#!/usr/bin/env python3
"""Generate only the final answer from a recorded case, without re-executing tools."""
import argparse
import json
from pathlib import Path

from opends_world.config import load_config
from opends_world.finalization import finalize, load_artifact, reconstruct_messages, replay_context
from opends_world.models import create_backend, initial_messages
from opends_world.runner import DEFAULT_PROMPT, _question_with_input_video
from opends_world.trace import TraceWriter, read_events
from opends_world.types import Sample
import hashlib


def source_context(trace_path, run_config):
    events, _ = read_events(trace_path)
    start = next(e for e in events if e['event'] == 'run_start')
    if 'initial_context_artifact' in start:
        initial = load_artifact(start['initial_context_artifact'])['messages']
    else:
        prompt = run_config.get('agent', {}).get('system_prompt', DEFAULT_PROMPT)
        expected = run_config.get('system_prompt_sha256')
        if not expected or hashlib.sha256(prompt.encode()).hexdigest() != expected:
            raise ValueError('Cannot reconstruct original system prompt; provide the original verified configuration')
        input_record = start['input']
        sample = Sample(str(start['sample_id']), input_record['question'], {}, trace_path.parent)
        if input_record.get('input_video'):
            sample.input_video_path = Path(input_record['input_video']['path'])
        initial = initial_messages(_question_with_input_video(sample),
            Path(input_record['video_path']) if input_record.get('video_path') else None,
            Path(input_record['audio_path']) if input_record.get('audio_path') else None,
            prompt, run_config.get('agent', {}).get('max_media_bytes', 100_000_000))
    return reconstruct_messages(trace_path, initial, run_config.get('agent', {})), start


def run_finalization(trace_path, source_config, model_config, output_dir, agent_config=None, snapshot=None,
                     apply_current_policy=False):
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in ('finalization_result.json', 'finalization_context.json', 'trace.jsonl')):
        raise ValueError(f'Recovery output already exists: {output_dir}')
    trace_path = Path(trace_path)
    expected_hash = None
    if snapshot:
        messages, config, reason = replay_context(Path(snapshot))
        expected_hash = json.loads(Path(snapshot).read_text()).get('request_sha256')
        if not expected_hash:
            raise ValueError('Snapshot lacks a request fingerprint; exact replay cannot be verified')
        if apply_current_policy:
            config = {**config, **(agent_config or {})}
            expected_hash = None
        start = {'sample_id': trace_path.stem, 'benchmark': source_config.get('benchmark', 'replay')}
    else:
        messages, start = source_context(trace_path, source_config)
        config = {**source_config.get('agent', {}), **(agent_config or {})}
        stops = [e for e in read_events(trace_path)[0] if e['event'] in {'research_stopped', 'forced_final'}]
        reason = stops[0].get('reason', 'replay') if stops else 'replay'
    common = {'sample_id': start['sample_id'], 'benchmark': start['benchmark'],
              'source_trace': str(trace_path), 'source_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest()}
    with TraceWriter(output_dir / 'trace.jsonl', common) as trace:
        trace.write('replay_start', discarded_incomplete_record_lines=read_events(trace_path)[1],
                    apply_current_policy=apply_current_policy,
                    source_snapshot_sha256=hashlib.sha256(Path(snapshot).read_bytes()).hexdigest() if snapshot else None)
        result = finalize(create_backend(model_config), messages, config, output_dir, trace, reason,
                          expected_request_sha256=expected_hash)
        trace.write('replay_end', result=result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True, type=Path)
    parser.add_argument('--source-run-config', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path, help='Current model connection and finalization settings')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--snapshot', type=Path, help='Use an existing finalization_context.json for exact same-stop replay')
    parser.add_argument('--apply-current-policy', action='store_true',
                        help='Explicitly apply current agent settings to a snapshot while preserving its completed evidence')
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_finalization(args.trace, json.loads(args.source_run_config.read_text()), config['model'],
                              args.output_dir, config.get('agent', {}), args.snapshot, args.apply_current_policy)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['finalization_status'] == 'complete' else 2)


if __name__ == '__main__':
    main()
