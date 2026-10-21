"""Merge a state-tuning checkpoint into the base model weights."""

from argparse import ArgumentParser

import torch


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True, help="base .pth checkpoint")
    parser.add_argument("--state_checkpoint", required=True,
                        help="checkpoint produced by --peft state")
    parser.add_argument("--output", required=True, help="path of the merged .pth")
    args = parser.parse_args()

    with torch.no_grad():
        weights = torch.load(args.base_model, map_location="cpu")
        state = torch.load(args.state_checkpoint, map_location="cpu")

        missing = [k for k in state if k not in weights]
        if missing:
            raise KeyError(f"{len(missing)} state key(s) not present in the base "
                           f"model, e.g. {missing[:5]}")
        weights.update(state)
        torch.save(weights, args.output)
    print(f"merged {len(state)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
