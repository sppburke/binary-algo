#!/usr/bin/env python3
"""Run independent jobs in freshly forked worker processes (used by the CPCV harnesses' --path-jobs).

    for i, out in run_forked(fn, specs, n_workers): ...        yields (index, fn(specs[index])) as jobs finish

Each job gets its own process forked from the caller at launch time, so it sees the caller's data copy-on-write and
nothing another job did: its result depends only on the caller's state and its spec, never on scheduling. At most
n_workers jobs run at once. A job that raises, or a worker that dies without a result (memory cap, signal), ends the
whole run with SystemExit after the other workers are stopped. The caller must not have started OpenMP threads
(for example a LightGBM fit) before the first fork.
"""
import multiprocessing as mp
import traceback
from multiprocessing.connection import wait


def _child(fn, spec, conn):
    try:
        out = (True, fn(spec))
    except BaseException:                           # the parent reports it and stops the run
        out = (False, traceback.format_exc())
    conn.send(out)
    conn.close()


def run_forked(fn, specs, n_workers):
    ctx = mp.get_context('fork')
    running, nxt = {}, 0                            # job index -> (process, read end)

    def stop_all():
        for p, rd in running.values():
            p.terminate()
        for p, rd in running.values():
            p.join()
            rd.close()
        running.clear()

    try:
        while nxt < len(specs) or running:
            while nxt < len(specs) and len(running) < n_workers:
                rd, wr = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_child, args=(fn, specs[nxt], wr))
                p.start()
                wr.close()                          # the parent keeps only the read end: EOF if the worker dies
                running[nxt] = (p, rd)
                nxt += 1
            ready = wait([rd for _, rd in running.values()])
            for i in [i for i, (_, rd) in running.items() if rd in ready]:
                p, rd = running.pop(i)
                try:
                    ok, payload = rd.recv()
                except EOFError:
                    p.join()
                    raise SystemExit(f'job {i}: the worker died without a result (exit code {p.exitcode})')
                finally:
                    rd.close()
                p.join()
                if not ok:
                    raise SystemExit(f'job {i} failed in its worker:\n{payload}')
                yield i, payload
    finally:
        stop_all()


def in_order(results):
    """Re-order (index, value) pairs that arrive in any order into index order, yielding each as soon as it is next."""
    held, want = {}, 0
    for i, v in results:
        held[i] = v
        while want in held:
            yield want, held.pop(want)
            want += 1
