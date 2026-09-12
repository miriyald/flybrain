"""Step 6 - point the fly's circuit at something that is not a smell.

The mushroom body is a classifier that happens to be wired for odours. If sparse expansion
coding really is a general trick, the circuit should not care what its input means.

So: feed it pixels. The expansion layer is still the measured connectome, the sparsening is
still k-winners-take-all, and learning is still dopamine-gated depression. The only new thing
is that the 55 inputs are now pixels instead of glomeruli.

The data is split three ways. Anything tuned is tuned on the validation split; the test split
is scored once, at the end. Tuning against the test set would make the final number a
description of this script rather than a prediction about new digits.

scikit-learn supplies the digits and the scoring. It does not train anything - every
prediction here comes out of the fly.

Run:  python 06_recognise_digits.py
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from flylab import digits, model
from flylab.circuit import Circuit
from flylab.classifier import FlyClassifier

SEED = 0
Split = tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]


def report_encoding(train_images: npt.NDArray[np.float64], pixels: npt.NDArray[np.int64], n_glomeruli: int) -> None:
    variance = train_images.var(axis=0)
    kept = variance[pixels].sum() / variance.sum()
    dropped = train_images.shape[1] - n_glomeruli
    print(f"--- fitting {train_images.shape[1]} pixels into {n_glomeruli} glomeruli ---")
    print(f"  the fly has {n_glomeruli} input channels; an 8x8 digit has {train_images.shape[1]} pixels")
    print(f"  dropping the {dropped} least informative keeps {kept:.4%} of the variance")
    print(f"  pixels with zero variance across all {len(train_images)} training digits: {int((variance == 0).sum())}")
    print("  (the border of a digit raster is blank, so this costs almost nothing)")


def report_both_dopamine_systems(circuit: Circuit, pixels: npt.NDArray[np.int64], train: Split, validate: Split) -> None:
    """Punishment alone wastes half the circuit.

    Each variant gets its own best depression rate. Scoring them at a shared rate would be
    rigged, because punishment alone prefers a far gentler one.
    """
    print("\n--- does using both dopamine systems help? (validation split, each at its own best rate) ---")
    rates = (0.001, 0.01, 0.05, 0.1, 0.25, 0.5)
    for label, reward in (("punishment only (PPL1)", False), ("punishment + reward (PPL1 + PAM)", True)):
        best, best_rate = 0.0, rates[0]
        for rate in rates:
            classifier = FlyClassifier.from_circuit(circuit, pixels, rate=rate).fit(*train)
            if not reward:
                classifier.avoid = np.ones_like(classifier.avoid)
            score = accuracy_score(validate[1], classifier.predict(validate[0]))
            best, best_rate = max((best, best_rate), (score, rate))
        print(f"  {label:<34} {best:.1%} at rate {best_rate:<6} {'#' * round(best * 40)}")


def report_sparsity_sweep(circuit: Circuit, pixels: npt.NDArray[np.int64], train: Split, validate: Split) -> None:
    print(f"\n--- how sparse should the code be? (validation split, {circuit.n_kc} Kenyon cells) ---")
    for sparsity in (0.02, 0.05, 0.09, 0.10, 0.15, 0.25):
        sweep = FlyClassifier.from_circuit(circuit, pixels, sparsity=sparsity).fit(*train)
        score = accuracy_score(validate[1], sweep.predict(validate[0]))
        marker = "  <- the value used for odours" if abs(sparsity - model.SPARSITY) < 1e-9 else ""
        print(f"  {sparsity:>5.0%}  k={round(sparsity * circuit.n_kc):>4}  {score:.1%}  {'#' * round(score * 40)}{marker}")


def report_confusion(matrix: npt.NDArray[np.int64]) -> None:
    print("\n--- confusion matrix on the test split (rows = true, columns = predicted) ---")
    print("      " + "".join(f"{digit:>5}" for digit in range(len(matrix))))
    for digit, row in enumerate(matrix):
        cells = "".join(f"{value:>5}" if value else "    ." for value in row)
        print(f"  {digit}   {cells}   {row[digit] / row.sum():>6.0%}")


def report_failures(images: npt.NDArray[np.float64], truth: npt.NDArray[np.int64], predicted: npt.NDArray[np.int64]) -> None:
    print("\n--- where it still goes wrong ---")
    for index in np.flatnonzero(predicted != truth)[:2]:
        print(f"\n  true {truth[index]}, predicted {predicted[index]}")
        for line in digits.render(images[index]):
            print(f"      {line}")


def report_overlap(codes: npt.NDArray[np.bool_], labels: npt.NDArray[np.int64], matrix: npt.NDArray[np.int64]) -> None:
    """Classes that share Kenyon cells are the ones the circuit confuses."""
    print("\n--- why: how much do the digit classes share Kenyon cells? ---")
    centres = np.array([codes[labels == digit].mean(axis=0) for digit in range(len(matrix))])
    similarity = centres @ centres.T / np.maximum(np.linalg.norm(centres, axis=1) ** 2, 1e-9)
    np.fill_diagonal(similarity, 0.0)
    for _ in range(3):
        first, second = divmod(int(np.argmax(similarity)), similarity.shape[1])
        confusions = matrix[first, second] + matrix[second, first]
        print(f"    {first} and {second}: {similarity[first, second]:.0%} shared   -> {confusions} actual confusions")
        similarity[first, second] = similarity[second, first] = 0.0


def main() -> None:
    print(__doc__)
    images, labels = load_digits(return_X_y=True)
    train_images, rest_images, train_labels, rest_labels = train_test_split(
        images, labels, test_size=0.4, random_state=SEED, stratify=labels
    )
    validate_images, test_images, validate_labels, test_labels = train_test_split(
        rest_images, rest_labels, test_size=0.5, random_state=SEED, stratify=rest_labels
    )
    circuit = Circuit.load()
    pixels = digits.live_pixels(train_images, keep=circuit.n_glomeruli)
    report_encoding(train_images, pixels, circuit.n_glomeruli)

    train = (digits.to_glomeruli(train_images, pixels), train_labels)
    validate = (digits.to_glomeruli(validate_images, pixels), validate_labels)
    test = (digits.to_glomeruli(test_images, pixels), test_labels)
    print(f"\n  train {len(train_labels)}   validate {len(validate_labels)}   test {len(test_labels)}")

    report_both_dopamine_systems(circuit, pixels, train, validate)
    report_sparsity_sweep(circuit, pixels, train, validate)

    classifier = FlyClassifier.from_circuit(circuit, pixels).fit(*train)
    predictions = classifier.predict(test[0])
    accuracy = accuracy_score(test_labels, predictions)
    print("\n--- final model, scored once on the held-out test split ---")
    print(f"  sparsity {classifier.sparsity:.0%} of {circuit.n_kc} Kenyon cells, depression rate {classifier.rate}, one pass")
    print(f"  TEST ACCURACY  {accuracy:.1%}   (chance would be {1 / classifier.n_classes:.0%})")

    matrix = confusion_matrix(test_labels, predictions)
    report_confusion(matrix)
    report_failures(test_images, test_labels, predictions)
    report_overlap(classifier.encode(train[0]), train_labels, matrix)

    memory = classifier.fit(*validate).save()
    print(f"\nsaved what it learned: {memory}  ({memory.stat().st_size / 1e3:.0f} KB)")
    print("  the wiring is in circuit.npz; this file is the memory laid on top of it")
    print("\nNext: python 07_draw_a_digit.py")


if __name__ == "__main__":
    main()
