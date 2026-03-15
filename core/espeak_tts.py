import subprocess
from typing import List


def ipa_batch(words: List[str]) -> List[str]:
    """
    Convert a list of words to IPA strings using espeak-ng via WSL.

    Requires espeak-ng installed inside WSL:
        wsl sudo apt-get install -y espeak-ng

    Args:
        words: List of words/phrases to convert

    Returns:
        List of IPA strings (one per input word)
    """
    input_text = "\n".join(words) + "\n"

    p = subprocess.run(
        ["wsl", "bash", "-lc", "espeak-ng -q --ipa"],
        input=input_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="replace")
        out = p.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WSL espeak-ng failed rc={p.returncode}\nSTDERR:\n{err}\nSTDOUT:\n{out}"
        )

    text = p.stdout.decode("utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


if __name__ == "__main__":
    words = ["board mon", "Borgmon", "Xaelthar"]
    print(ipa_batch(words))
