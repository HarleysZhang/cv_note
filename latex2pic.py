import re
from urllib.parse import quote

if __name__ == "__main__":
    text = open("深度学习/神经网络压缩算法总结.md",encoding="utf-8").read()

    parts = text.split("$$")

    for i, part in enumerate(parts):
        if i & 1:
            parts[i] = f'![](http://latex.codecogs.com/gif.latex?{quote(part.strip())})'

    text_out = "\n\n".join(parts)

    lines = text_out.split('\n')
    for lid, line in enumerate(lines):
        parts = re.split(r"\$(.*?)\$", line)
        for i, part in enumerate(parts):
            if i & 1:
                parts[i] = f'![](http://latex.codecogs.com/gif.latex?{quote(part.strip())})'
        lines[lid] = ' '.join(parts)
    text_out = "\n".join(lines)

    with open("./深度学习/神经网络压缩算法总结new.md", "w", encoding='utf-8') as f:
        f.write(text_out)