"""Helper file for working with COSYScript (.fox) files."""

import typing
import subprocess
import os
import numpy as np
import numpy.typing as npt

__current_eval_ids = set()
__cosy_cwd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./eval/")


def content_subs(content: str, subs: dict[str, typing.Any]) -> str:
    """
    Substitute data into a COSYScript source file.

    Replaces `{{`keys`}}` with values from the subs dict.
    """
    for sub_from, sub_to in subs.items():
        content = content.replace(f"{{{{{sub_from}}}}}", str(sub_to))
    return content


def eval_fox(
    content: str, use_gui=False, main_fn_name: str | None = None
) -> typing.Callable[[], str]:
    """
    Executes a COSYScript file.

    [note that the script's cwd will be this directory (`cosy/eval/`)]

    To print to the output, use file handle 0.

    Returns a callable, which when called waits for the script to
    finish.
    """
    vars = ""
    content = content.strip()
    while content.upper().startswith("VARIABLE "):
        i = content.index(";") + 1
        vars += "\n" + content[:i]
        content = content[i:].strip()

    eval_id = 0
    while eval_id in __current_eval_ids:
        eval_id += 1
    __current_eval_ids.add(eval_id)

    output_file = os.path.join(__cosy_cwd, f"./{eval_id}.txt")

    with open(output_file, "w", encoding="utf8") as f:
        pass
    content = (
        f"""
        INCLUDE 'COSY';
        PROCEDURE run;
            {vars}
            {content}
            OPENF 0 './{eval_id}.txt' 'REPLACE';
            {main_fn_name if main_fn_name is not None else ("main_gui" if use_gui else "main")};
            CLOSEF 0;
        ENDPROCEDURE;
        run;
        END;
    """.strip()
        + "\n"
    )

    with open(os.path.join(__cosy_cwd, f"./{eval_id}.fox"), "w", encoding="utf8") as f:
        f.write(content)
    cosy_process = subprocess.Popen(
        (
            [
                os.path.join(__cosy_cwd, "./cosy"),
                f"{eval_id}.fox",
            ]
            if not use_gui
            else [
                "java",
                "-jar",
                "./COSYGUI.jar",
                f"{eval_id}.fox",
            ]
        ),
        stdout=subprocess.DEVNULL,
        cwd=__cosy_cwd,
    )

    def cosy_proc_join():
        cosy_process.wait()
        __current_eval_ids.remove(eval_id)
        with open(output_file, "r", encoding="utf8") as f:
            return f.read()

    return cosy_proc_join


def read_sub_eval(
    location: str,
    subs: dict[str, typing.Any],
    use_gui=False,
    main_fn_name: str | None = None,
) -> typing.Callable[[], str]:
    """
    Read and execute the given COSYScript (.fox) file with the
    given substitutions applied (using `content_subs`).
    """
    with open(location, "r", encoding="utf8") as f:
        return eval_fox(content_subs(f.read(), subs), use_gui, main_fn_name)


def parse_transfer_map(
    src: str,
) -> tuple[typing.Callable[[npt.NDArray], npt.NDArray], str]:
    """
    Read in a cosy transfer map, returning the function that it represents.

    `(x,a,y,b,t,K) -> (x,a,y,b,t)`
    """
    src_ = src.strip()
    i = src_.find("----")
    if i == 0:
        while i < len(src) and src[i] != "\n":
            i += 1
        src = src_[i + 1 :]
    src_end_i = src.find("----")
    if src_end_i == -1:
        raise ValueError("map data invalid")
    src, rem = src[:src_end_i], src[src.find("\n", src_end_i) :][1:]

    map_data = [
        (
            np.array(
                [
                    float(line[1 + 0 * 14 : 1 + 1 * 14]),
                    float(line[1 + 1 * 14 : 1 + 2 * 14]),
                    float(line[1 + 2 * 14 : 1 + 3 * 14]),
                    float(line[1 + 3 * 14 : 1 + 4 * 14]),
                    float(line[1 + 4 * 14 : 1 + 5 * 14]),
                ]
            ),
            [int(v) for v in line[2 + 5 * 14 :]],
        )
        for line in src.splitlines()
        if line.strip() != ""
    ]

    def eval_map(args: npt.ArrayLike):
        return sum(
            row * np.prod(np.power(args, basis_powers))
            for row, basis_powers in map_data
        ) + np.zeros(5)

    return eval_map, rem


def parse_write(src: str) -> tuple[str, npt.NDArray, str]:
    """
    Read in the output of a COSYScript WRITE.
    """
    src = src.strip().replace("\r", "")
    i = src.find("\n")
    k = src[:i]
    v = []
    i_end = i
    while len(src) > i + 1 and src[i + 1] == " ":
        i_end = src.find("\n", i + 1)
        v.append(float(src[i:i_end]))
        i = i_end
    rem = src[i_end + 1 :] if i_end > 0 else ""
    return k, np.array(v), rem


def parse_write_dict(
    src: str, reduce_single=True
) -> tuple[dict[str, npt.NDArray], str]:
    """
    Read in the output of several COSYScript WRITE calls, stopping at EOF or an empty line.
    """
    src = src.strip()
    out = {}
    while len(src) > 0 and src[0] != "\n":
        k, v, src = parse_write(src)
        out[k] = v[0] if v.size == 1 and reduce_single else v
    return out, src


with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "./utils.fox"),
    "r",
    encoding="utf8",
) as f:
    INCLUDE_UTILS = {"INCLUDE_UTILS": f.read()}
