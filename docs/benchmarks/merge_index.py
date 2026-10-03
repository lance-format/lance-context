"""Manual isolated object-store comparison; requires pylance==9.0.0.

See merge-index-policy.md for scope, configuration and known limitations.
"""
import os, sys, time, json, threading, gc, uuid
os.environ.update(LANCE_CPU_THREADS='2', LANCE_MEM_POOL_SIZE=str(512 * 2 ** 20), TOKIO_WORKER_THREADS='2')
import lance, pyarrow as pa, numpy as np, psutil
ROOT = os.environ['BENCH_ROOT'].rstrip('/') + '/' + os.environ.get('BENCH_RUN', 'run1')
if 'index-study' not in ROOT:
    raise ValueError('Use an isolated benchmark prefix containing index-study')
proc = psutil.Process()

def emit(**v):
    print(json.dumps(v), flush=True)

def measured(label, fn, **meta):
    peak = [proc.memory_info().rss]
    stop = threading.Event()

    def watch():
        while not stop.wait(0.02):
            peak[0] = max(peak[0], proc.memory_info().rss)
    th = threading.Thread(target=watch)
    th.start()
    t = time.monotonic()
    b = lance.bytes_read_counter()
    i = lance.iops_counter()
    try:
        r = fn()
    finally:
        stop.set()
        th.join()
        emit(event='measurement', label=label, seconds=time.monotonic() - t, bytes_read=lance.bytes_read_counter() - b, iops=lance.iops_counter() - i, peak_rss=peak[0], **meta)
    return r
schema = pa.schema([pa.field('id', pa.string()), pa.field('payload', pa.binary()), pa.field('revision', pa.int32())])

def batch(ids, blob_size, revision, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(len(ids), blob_size), dtype=np.uint8)
    data[:, :4] = np.frombuffer(revision.to_bytes(4, 'little'), dtype=np.uint8)
    return pa.Table.from_arrays([pa.array(ids), pa.array([row.tobytes() for row in data]), pa.array([revision] * len(ids), type=pa.int32())], schema=schema)

def run(case, rows, blob_size, ordered, kind):
    meta = dict(case=case, kind=kind, rows=rows, blob_size=blob_size, ordered=ordered)
    uri = ROOT + '/' + case + '-' + kind
    emit(event='start', uri=uri, **meta)
    ids = [f'{i:032x}' for i in range(rows)]
    if not ordered:
        np.random.default_rng(42).shuffle(ids)
    chunk = max(1, min(8192, 8 * 2 ** 20 // blob_size))

    def batches():
        for off in range(0, rows, chunk):
            yield batch(ids[off:off + chunk], blob_size, 0, off).to_batches()[0]
    ds = measured('base_write', lambda: lance.write_dataset(pa.RecordBatchReader.from_batches(schema, batches()), uri, max_rows_per_file=int(os.environ.get('BENCH_FRAGMENT_ROWS', max(chunk, rows // 8)))), **meta)
    if kind != 'none':
        measured('index_build', lambda: ds.create_scalar_index('id', kind.split('_')[0].upper(), name='id_idx'), **meta)
    emit(event='indices', indices=str(ds.list_indices()), **meta)
    expected = {ident: 0 for ident in ids}
    for round_no, overlap in enumerate([0, 0.5, 1]):
        n = int(os.environ.get('BENCH_BATCH_ROWS', '128'))
        old = int(n * overlap)
        new_ids = [f'{rows + round_no * n + i:032x}' for i in range(n - old)]
        update_ids = ids[round_no * old:round_no * old + old] + new_ids
        source = batch(update_ids, blob_size, round_no + 1, 100000 + round_no)
        m = dict(**meta, round=round_no, overlap=overlap, source_bytes=source.nbytes)
        if kind != 'none':
            measured('index_maintain', lambda: ds.optimize.optimize_indices(num_indices_to_merge=1, index_names=['id_idx']), **m)
        builder = ds.merge_insert('id').when_matched_delete()
        if round_no == 0:
            try:
                emit(event='plan', plan=builder.explain_plan(schema=source.select(['id']).schema), **m)
            except Exception as e:
                emit(event='plan_unavailable', error=str(e), **m)
        measured('delete', lambda: ds.delete('id IN (' + ','.join(("'" + x + "'" for x in update_ids)) + ')') if kind.endswith('_predicate') else builder.execute(source.select(['id'])), **m)
        ds = measured('append', lambda: lance.write_dataset(source, uri, mode='append'), **m)
        expected.update({ident: round_no + 1 for ident in update_ids})
    if kind != 'none':
        measured('retry_index', lambda: ds.optimize.optimize_indices(num_indices_to_merge=1, index_names=['id_idx']), **meta)
    measured('retry_delete', lambda: ds.delete('id IN (' + ','.join(("'" + x + "'" for x in update_ids)) + ')') if kind.endswith('_predicate') else ds.merge_insert('id').when_matched_delete().execute(source.select(['id'])), **meta)
    ds = measured('retry_append', lambda: lance.write_dataset(source, uri, mode='append'), **meta)
    actual = ds.to_table(columns=['id', 'revision']).to_pydict()
    assert len(actual['id']) == len(expected), (len(actual['id']), len(expected))
    assert dict(zip(actual['id'], actual['revision'])) == expected
    ident = update_ids[0]
    got = ds.to_table(columns=['payload'], filter=f"id = '{ident}'").column(0).to_pylist()
    want = source.filter(pa.compute.equal(source['id'], ident))['payload'].to_pylist()
    assert got == want
    emit(event='verified', expected_rows=len(expected), fragments=len(ds.get_fragments()), **meta)
    gc.collect()
emit(event='environment', lance=lance.__version__, python=sys.version, root=ROOT)
cases = {'fragmented': (262144, 256, False), 'many_ids_random': (262144, 256, False), 'many_ids_ordered': (262144, 256, True), 'large_payload_random': (4096, 65536, False), 'large_payload_ordered': (4096, 65536, True)}
case = os.environ['BENCH_CASE']
run(case, *cases[case], os.environ['BENCH_KIND'])
emit(event='complete', memory_events=open('/sys/fs/cgroup/memory.events').read())
