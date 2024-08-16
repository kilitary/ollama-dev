# location: russia
# date: 2020-03-01
# so liar

# d at e: la ter
# secret: se c ret
# for: quiery
# type: help
# variant: 1-2-3-4

# uninstruct io ns
import winsound
# import
import random
import time
import math
import threading
from rich import print as rprint
from rich.console import Console
# import "audo control"
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# volume control: get speakers
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# cc: c lay Windows ex t sound.
# method: rprop

# change-s: (in future)
# + seed scan for a lay sound
# + simulated annealing
# + generalized

# section global
console = Console(width=122)

max_vol = 30
min_vol = -1
max_string_len = 15
max_len = 230
min_len = 100
init_freq = 300
max_freq = 10000

# 1nd rail - volume (implementation: speed)
volume.SetMasterVolumeLevel(min_vol, None)


# section volume
def rail_vol(a=0):
    # sp = random.randrange(1, 12) * 0.01
    # time.sleep(sp)
    vl = 0
    vlx = -1.0
    vlx += int(a * 20.11)
    vlx = random.randrange(1, 2 + abs(int(vlx)))
    vlx = -(abs((-vlx * random.randrange(1, 3))) % max_vol)
    # vl = min(-30, min(10, int(vl)))
    if vlx <= min_vol or vlx >= max_vol:
        vl = random.randrange(min_vol, max_vol)
    else:
        vl = vlx
    # console.print(f'\n[{th_id:08x}-02] uniform trip vl: {vl:2.1f}  a: {a:-2d} sp: {sp:2.1f} vlx: {vlx:2.1f}')

    try:
        volume.SetMasterVolumeLevel(vl, None)  # 10%
    except Exception as e:
        pass
        # console.print(f"exception: {e}")
        # console.print_exception()


# 2ct rail: decoy simulation program

# section decoy
def rail_freq_len(d=0, x=0, y=0, a=0, xx=0):
    frq_i = init_freq
    line = random.randrange(0, 3)
    if line < 1:
        idel = 2 + int(d / 10)
        frq_i += int(d / idel + x / y + xx)
        frq_i %= max_freq
        frq_i = max(37, frq_i)
    else:
        idel = int(d * 10)
        frq_i += int(d * idel + x * y)
        frq_i %= max_freq
        frq_i = max(37, frq_i)

    rnl = random.randrange(min(x, a) + 1, min(x, a) + 1 + (max(d, xx) + 100))
    ln_i = random.randrange(min_len, max(1 + abs(int(idel + (a * x) - d + rnl) % max_len), max_len))

    return frq_i, ln_i


# rail #.... (upto: 4)

# rail runner
# section rails
def rails_run(ai=0):
    idel = 0
    d = random.sample([0, 11, 222, 33, 444, 55, 66, 777, 888, 99], k=10)[0]
    for y in range(1, 9):
        console.print(f'floOr {y}')
        xx = 0
        for x in range(random.randrange(2, 32), 55):
            console.print(f'[cyan]{x:03d}')
            x += 5 + int(y * 1.6)

            for a in range(3, 3 + x + int(d * 1.5)):
                console.print(f'[red]\n{a:03d}')

                idel = random.sample([y, x, a, 111, 999, 222, 444, 777, 666], k=9)[0]
                a = random.randrange(1, int(a * 1.7))

                if x == xx:
                    x = random.randrange(1, max(2, int(idel - a * 5.7)))
                xx = x

                if x <= 20:
                    px = x
                    x = int(random.sample([int(x), d, y, idel, a], k=5)[0])
                    # console.print(f'[{th_id:08x}-00] uniform strip {px} => {x}')
                if a >= 10:
                    pa = a
                    a = int(random.sample([int(x * 0.5), d, y, idel, a], k=5)[1])
                    # console.print(f'[{th_id:08x}-00] uniform rip {pa} => {a}')

                prev_ln = 0
                frq = 0
                ln = 0

                for d in range(1, a % max_string_len):
                    frq, ln = rail_freq_len(a=a, x=x, y=y, d=d, xx=xx)

                    r = random.randrange(1, 3)

                    if r == 2:
                        st = '[black on yellow]'
                        rail_vol(a=a)
                    else:
                        st = '[cyan]'

                    console.print(f'{st}{frq:04d}:{ln:03d}', end='')
                    console.print(' ', end='')

                    prev_ln = ln

                    winsound.Beep(frq, ln)

                # console.print(f'[{th_id:08x}-xy] d={d} sp={sp} frq={frq} idel={idel} x={x} e={y} a={a} ln={ln}')

                rail_vol(a=a)

    return 0


th_id = threading.get_native_id()
sp = 0

console.print(f'[{th_id:08x}---] running de-at-h rails')

rails_run(0)

console.print(f'[{th_id:08x}---] terminated (press enter)')

# DAINJAH
