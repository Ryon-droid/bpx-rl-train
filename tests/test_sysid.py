"""Identification regression tests: coordinate/time mistakes and physics replay."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpx_sysid.core import (SDK_NAMES, JOINT_NAMES, DEFAULT_POSITION, ROOT, model_contract,
                            parameter_vector, parameters, reorder, trajectory)
from bpx_sysid.data import load_dataset, prepare, save_dataset
from bpx_sysid.capture import snapshot, collect
from bpx_sysid.optimize import bounds_matrix


class CoordinatesAndDataTests(unittest.TestCase):
    def test_sdk_mapping_roundtrip_and_invalid_duplicate_names(self):
        encoded = np.arange(12.)
        policy = reorder(encoded, SDK_NAMES)
        np.testing.assert_array_equal(policy, [0,3,6,9,1,4,7,10,2,5,8,11])
        np.testing.assert_array_equal(reorder(policy, JOINT_NAMES, SDK_NAMES), encoded)
        with self.assertRaises(ValueError):
            reorder(encoded, [SDK_NAMES[0]]*12)

    def test_trajectory_continuity_limits_and_excessive_chirp_rejected(self):
        t,q,checks = trajectory(duration=4., ramp=1.)
        np.testing.assert_allclose(np.diff(t), .005, atol=1e-12)
        np.testing.assert_allclose(q[0], DEFAULT_POSITION)
        np.testing.assert_allclose(q[-1], DEFAULT_POSITION)
        self.assertLess(checks['peak_speed_rad_s'], .5)
        with self.assertRaises(ValueError):
            trajectory(f1=10., amplitude=.5)

    def test_delay_steps_seconds_and_urdf_provenance_are_checked(self):
        a = np.r_[np.full(12,.003), np.zeros(36),3.]
        p = parameters(a, model_contract())
        self.assertEqual(p['delay_s'], .015)
        p['delay_s'] = .0075
        with self.assertRaises(ValueError):
            parameter_vector(p)
        p = parameters(a, model_contract())
        p['urdf_sha256'] = 'other'
        with self.assertRaises(ValueError):
            parameter_vector(p)

    def test_readonly_sdk_does_not_construct_joint_controller_or_send_commands(self):
        sdk = Mock()
        state = sdk.RequestRobotState.return_value
        state.connect.return_value = True
        state.getJointStateTimestamp.return_value = 100
        for name in ('getJointPosition','getJointVelocity','getJointTorque'):
            getattr(state,name).return_value = [0.]*12
        sdk.__version__ = 'fake'
        with tempfile.TemporaryDirectory() as d:
            report=collect('unused', Path(d)/'raw.jsonl', seconds=.025, rate=100., sdk=sdk)
        sdk.JointLevelControl.assert_not_called()
        state.setJointCommand.assert_not_called()
        state.disconnect.assert_called_once()
        self.assertEqual(report['commands_sent'],0)
        self.assertEqual(report['unique_frames'],1)
        state.getJointStateTimestamp.side_effect = [100, 101]
        self.assertIsNone(snapshot(state))

    def raw_session(self):
        c=model_contract()
        rows=[dict(kind='metadata', source='hardware', session_id='fixture-1', time_basis='host_monotonic_ns',
                   joint_names=list(SDK_NAMES), sequence_kind='packet_counter',
                   capture_semantics='sequence_bracketed_cache')]
        for i in range(9):
            ns=10**15+i*10_000_000
            rows.append(dict(kind='feedback',seq=i,host_monotonic_ns=ns,
                q=[float(i)]*12,dq=[0.]*12,tau=[0.]*12))
            rows.append(dict(kind='command',host_monotonic_ns=ns,sent_ok=True,
                kp=[30.]*12,kd=[1.]*12,q_des=[float(i)]*12,dq_des=[0.]*12,tau_ff=[0.]*12))
        return c,rows

    def test_command_zero_order_hold_and_feedback_time_are_preserved(self):
        c,rows=self.raw_session()
        # Duplicate cache reads must not become extra measurements.
        rows.insert(3,dict(rows[1]))
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.jsonl';raw.write_text('\n'.join(json.dumps(x) for x in rows))
            prepare(raw,Path(d)/'prepared',c)
            data,meta=load_dataset(Path(d)/'prepared')
            np.testing.assert_allclose(data['dof_pos'][1],.5)
            np.testing.assert_allclose(data['des_dof_pos'][1],0.)
            self.assertEqual(meta['duplicate_frames'],1)
            datafile=Path(d)/'prepared/data.npz'
            datafile.write_bytes(datafile.read_bytes()+b'changed')
            with self.assertRaises(ValueError): load_dataset(Path(d)/'prepared')

    def test_raw_dropout_unknown_inputs_and_readonly_data_cannot_be_fitted(self):
        for mistake in ('dropout','gain','feedforward','readonly'):
            c,rows=self.raw_session()
            if mistake=='dropout':
                rows=[x for x in rows if x.get('seq') not in (1,2,3,4)]
            elif mistake=='gain': rows[2]['kp']=[31.]*12
            elif mistake=='feedforward': rows[2]['tau_ff']=[1.]*12
            else: rows[0]['source']='read_only'
            with tempfile.TemporaryDirectory() as d:
                raw=Path(d)/'raw.jsonl';raw.write_text('\n'.join(json.dumps(x) for x in rows))
                with self.assertRaises(ValueError, msg=mistake): prepare(raw,Path(d)/'prepared',c)

    def test_synthetic_bounds_cannot_be_used_on_hardware(self):
        path=ROOT/'calibration/templates/synthetic-bounds.json'
        b,_=bounds_matrix(path,True)
        self.assertEqual(b.shape,(49,2))
        with self.assertRaises(ValueError): bounds_matrix(path,False)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo not installed')
class FixedBasePhysicsTests(unittest.TestCase):
    def test_truth_replays_better_than_wrong_delay_and_bias(self):
        from bpx_sysid.mujoco_backend import MuJoCoReplay
        t,target,_=trajectory(duration=4.,ramp=1.,joint='all')
        contract=model_contract()
        sim=MuJoCoReplay(contract)
        truth=np.r_[np.full(12,.006),np.full(12,.03),np.full(12,.01),np.linspace(-.02,.02,12),4.]
        d=dict(dof_pos=target.copy(),dof_vel=np.zeros_like(target),des_dof_pos=target)
        self.assertEqual(sim.model.nv,12)
        d['dof_pos']=sim.replay(truth,d)
        exact=sim.replay(truth,d)
        np.testing.assert_allclose(exact,d['dof_pos'],atol=1e-12)
        wrong=truth.copy();wrong[-1]=0;wrong[36:48]=0
        scores,_=sim.evaluate([truth,wrong],d,100)
        self.assertEqual(scores[0],0.)
        self.assertGreater(scores[1],1e-7)
        self.assertEqual(sim.data.ncon,0)
        with tempfile.TemporaryDirectory() as out:
            save_dataset(Path(out)/'data',t,d['dof_pos'],target,contract,velocity=d['dof_vel'],source='synthetic')
            loaded,meta=load_dataset(Path(out)/'data')
            self.assertEqual(meta['source'],'synthetic')
            np.testing.assert_allclose(loaded['dof_pos'],d['dof_pos'])


    @unittest.skipUnless(importlib.util.find_spec('cmaes'), 'Optional CMA-ES not installed')
    def test_fit_export_is_an_actual_independently_replayed_candidate(self):
        from bpx_sysid.mujoco_backend import MuJoCoReplay
        from bpx_sysid.optimize import fit
        t,target,_=trajectory(duration=4.,ramp=1.,joint='all')
        c=model_contract()
        model=MuJoCoReplay(c)
        p=np.r_[np.full(12,.006),np.full(12,.03),np.full(12,.02),np.zeros(12),3.]
        d=dict(time=t,dof_pos=target.copy(),dof_vel=np.zeros_like(target),des_dof_pos=target)
        d['dof_pos']=model.replay(p,d)
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            save_dataset(base/'data',t,d['dof_pos'],target,c,velocity=d['dof_vel'],
                         source='synthetic',source_session_id='unique-test-session')
            data,meta=load_dataset(base/'data')
            report=fit(model,data,meta,ROOT/'calibration/templates/synthetic-bounds.json',
                       base/'fit',population=4,generations=1)
            saved=json.loads((base/'fit/best.json').read_text())
            score,_=model.evaluate([parameter_vector(saved)],data,100)
            self.assertAlmostEqual(float(score[0]),report['best']['mse_rad2'],places=14)
            self.assertEqual(saved['training_session_id'],'unique-test-session')
            self.assertEqual(saved['training_data_sha256'],meta['data_sha256'])
            self.assertFalse(report['holdout_validated'])
            self.assertTrue((base/'fit/mean_replay.npy').exists())


if __name__ == '__main__': unittest.main()
