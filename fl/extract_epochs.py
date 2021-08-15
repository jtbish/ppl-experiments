import sys


def main():
    base_dir = sys.argv[1]
    num_to_extract = int(sys.argv[2])
    assert num_to_extract >= 1
    avg_time_steps_txt = f"{base_dir}/avg_time_steps.txt"
    print(f"extracting {num_to_extract} epochs from {avg_time_steps_txt}")
    with open(avg_time_steps_txt, "r") as fp:
        lines = fp.readlines()

    num_to_extract = min(num_to_extract, len(lines))
    total = 0
    epochs = []
    for idx in range(0, num_to_extract):
        total += int(lines[idx])
        epochs.append(total)
    print(epochs)

    epochs_str = ",".join([str(e) for e in epochs])
    with open(f"{base_dir}/epochs.txt", "w") as fp:
        fp.write(epochs_str)


if __name__ == "__main__":
    main()
