#  Copyright (c) 2025. kilitary@gmail.com

import os
import glob
import requests
import time
import rich
from rich import print
from rich import print_json
import sys
import json
import xml.dom.minidom
import xml
import re
import random
import pefile
import shutil

links = requests.get('https://www.nirsoft.net/pad/pad-links.txt')
links = links.text.split('\n')
updated = 0
total = len(links)
prevs = glob.glob(r'h:\*')
for prev in prevs:
    print(f'unlink {prev}')
    try:
        shutil.rmtree(prev)
    except Exception as e:
        pass
tot = 0.
for link in links:
    if link == '':
        print(f'done with all, updated={updated}')
        break
    updated += 1

    xm = requests.get(link)
    xm = xm.text
    xm = json.dumps(xm)
    url = re.findall(r'Primary_Download_URL>(.*?)</Prim', xm)

    while True:
        try:
            resp = requests.get(url[0], headers={
                'Referer': 'https://www.nirsoft.net/pad/index.html',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4'
            })
        except Exception as e:
            print(f'[red]failed with {url[0]}: {e}')
            time.sleep(5)
            continue
        break

    if resp.status_code != 200:
        print(f'[red]failed with {url[0]}: resp.status_code={resp.status_code}')
        continue

    data = resp.content

    rand_name = str(random.randrange(10, 99120)) + '.zip'
    dr = str(random.randrange(1, 100000))
    os.makedirs(rf'h:\\\{dr}')
    drr = os.path.join(r'h:\\', dr)
    path = os.path.join(r'h:\\', dr, rand_name)

    with open(path, 'wb') as f:
        print(f'writing {len(data)} bytes to {path}: ')
        ret = f.write(data)
        print(f'wrote {ret} bytes\n')

    sz = f'{os.path.getsize(path) / 1024.0 / 1024.0:.2f} MB'
    tot += float(os.path.getsize(path) / 1024.0 / 1024.0)
    print(f'get {link} -> [{updated:d}/{total:d}]\r\n({(updated / total) * 100.0:.2f}%) size: {sz} of {tot:.2f} total MB')

    os.chdir(drr)

    print(f'unpacking {path} ...')
    os.system(f'7z x -p0 -bb0 {path} > o')

    # find exe
    print(f'url={url}')

    try:
        exepath = re.findall(r'\w+/\w+/([^.]{1,25})\.zip', url[0])
        name = exepath[0]
        rich.print(f'name={name}')
        exes = glob.glob(f'*.exe')

        epath = os.path.join(drr, exes[0])
        print(f'found {epath} drr={drr}')
    except Exception as e:
        print(f'EXE is broken or infected: {e}')
        continue

    try:
        pe = pefile.PE(epath)
    except Exception as e:
        print(f'POSSIIBLY PASSSWORDED')
        continue
    # determine x64

    print(f'machine={pe.FILE_HEADER.Machine} ({pe.FILE_HEADER.Machine:x})')
    # unpack again to proper nirsoft dir

    if pe.FILE_HEADER.Machine == 0x8664:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\x64\{name}')
    elif pe.FILE_HEADER.Machine == 0x14c:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\{name}')
    else:
        print(f'unknown machine {pe.FILE_HEADER.Machine} ({pe.FILE_HEADER.Machine:x})')
        time.sleep(10)

    print(f'[cyan]unpacking {path} to {dest}...')
    os.system(rf'7z x -p0 -y -bb0 {path} -o{dest} > o')
    pe.close()
    os.chdir(r"h:\\")
    shutil.rmtree(drr)
    print(f'done with {name}')
