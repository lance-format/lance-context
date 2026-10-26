"""Finite fixture's loopback HTTP/2 relay; drop an armed response DATA frame.

The caller verifies the transaction persisted through a separate direct client.
Frame position alone is not accepted as evidence of a lost transaction ack.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen-port', type=int, required=True)
    parser.add_argument('--upstream-port', type=int, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    args = parser.parse_args()
    arm = args.state_dir / 'drop-arm.json'
    evidence = args.state_dir / 'drop-evidence.json'

    async def connection(reader, writer):
        upstream_reader, upstream_writer = await asyncio.open_connection('127.0.0.1', args.upstream_port)
        remaining = None
        mode = 'drop'

        async def requests():
            while data := await reader.read(65536):
                upstream_writer.write(data)
                await upstream_writer.drain()

        async def responses():
            nonlocal remaining, mode
            while True:
                header = await upstream_reader.readexactly(9)
                length = int.from_bytes(header[:3], 'big')
                body = await upstream_reader.readexactly(length)
                if header[3] == 0:  # HTTP/2 DATA; SETTINGS/HEADERS are untouched.
                    if remaining is None and arm.exists():
                        armed = json.loads(arm.read_text())
                        remaining = armed['data_frame']
                        mode = armed.get('mode', 'drop')
                        arm.unlink()
                    if remaining is not None:
                        remaining -= 1
                        if remaining == 0:
                            evidence.write_text(json.dumps({'at': time.time(), 'stream_id': int.from_bytes(header[5:], 'big'), 'data_bytes': length, 'mode': mode}))
                            if mode == 'drop':
                                return
                            assert mode == 'hold'
                            while not (args.state_dir / 'drop-release').exists():
                                await asyncio.sleep(.01)
                            (args.state_dir / 'drop-release').unlink()
                            remaining = None
                writer.write(header + body)
                await writer.drain()

        tasks = [asyncio.create_task(requests()), asyncio.create_task(responses())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            upstream_writer.close()
            await asyncio.gather(writer.wait_closed(), upstream_writer.wait_closed(), return_exceptions=True)

    server = await asyncio.start_server(connection, '127.0.0.1', args.listen_port)
    async with server:
        await server.serve_forever()


asyncio.run(main())
