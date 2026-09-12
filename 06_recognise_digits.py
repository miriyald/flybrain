"""Step 6 - point the fly's circuit at something that is not a smell.

The mushroom body is a classifier that happens to be wired for odours. If sparse expansion
coding really is a general trick, the circuit should not care what its input means.

So: feed it pixels. The expansion layer is still the measured connectome, the sparsening is
still k-winners-take-all, and the learning rule is still dopamine-gated depression. The only
new thing is that the 55 inputs are now pixels instead of glomeruli.

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


def report_encoding(train_images: npt.NDArray[np.float64], pixels: npt.NDArray[np.int64], n_glomeruli: int) -> None:
    variance = train_images.var(axis=0)
    kept = variance[pixels].sum() / variance.sum()
    dropped = train_images.shape[1] - n_glomeruli
    print(f"--- fitting {train_images.shape[1]} pixels into {n_glomeruli} glomeruli ---")
    print(f"  the fly has {n_glomeruli} input channels; an 8x8 digit has {train_images.shape[1]} pixels")
    print(f"  dropping the {dropped} least informative keeps {kept:.4%} of the variance")
    print(f"  pixels with zero variance across all {len(train_images)} training digits: {int((variance == 0).sum())}")
    print("  (the border of a digit raster is blank, so this costs almost nothing)")


def report_confusion(matrix: npt.NDArray[np.int64]) -> None:
    print("\n--- confusion matrix (rows = true, columns = predicted) ---")
    print("      " + "".join(f"{digit:>5}" for digit in range(len(matrix))))
    for digit, row in enumerate(matrix):
        cells = "".join(f"{value:>5}" if value else "    ." for value in row)
        print(f"  {digit}   {cells}   {row[digit] / row.sum():>6.0%}")


def report_failures(images: npt.NDArray[np.float64], truth: npt.NDArray[np.int64], predicted: npt.NDArray[np.int64]) -> None:
    print("\n--- where it goes wrong ---")
    for index in np.flatnonzero(predicted != truth)[:3]:
        print(f"\n  true {truth[index]}, predicted {predicted[index]}")
        for line in digits.render(images[index]):
            print(f"      {line}")


def report_overlap(codes: npt.NDArray[np.bool_], labels: npt.NDArray[np.int64], matrix: npt.NDArray[np.int64]) -> None:
    """Classes that share Kenyon cells are the ones the circuit confuses."""
    print("\n--- why: how much do the digit classes share Kenyon cells? ---")
    centres = np.array([codes[labels == digit].mean(axis=0) for digit in range(len(matrix))])
    similarity = centres @ centres.T / np.maximum(np.linalg.norm(centres, axis=1) ** 2, 1e-9)
    np.fill_diagonal(similarity, 0.0)
    print("  most-confusable pairs by shared code:")
    for _ in range(4):
        first, second = divmod(int(np.argmax(similarity)), similarity.shape[1])
        confusions = matrix[first, second] + matrix[second, first]
        print(f"    {first} and {second}: {similarity[first, second]:.0%} shared   -> {confusions} actual confusions")
        similarity[first, second] = similarity[second, first] = 0.0


def report_sparsity_sweep(circuit: Circuit, train: tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]], test: tuple) -> None:
    print(f"\n--- does the fly's own sparsity suit pixels? ({circuit.n_kc} Kenyon cells) ---")
    for sparsity in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
        sweep = FlyClassifier.from_circuit(circuit, sparsity=sparsity).fit(*train)
        score = accuracy_score(test[1], sweep.predict(test[0]))
        marker = "  <- the value used for odours" if abs(sparsity - model.SPARSITY) < 1e-9 else ""
        print(f"  {sparsity:>5.0%}  k={round(sparsity * circuit.n_kc):>4}  {score:.1%}  {'#' * round(score * 40)}{marker}")


def main() -> None:
    print(__doc__)
    images, labels = load_digits(return_X_y=True)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    circuit = Circuit.load()

    pixels = digits.live_pixels(train_images, keep=circuit.n_glomeruli)
    report_encoding(train_images, pixels, circuit.n_glomeruli)

    train = digits.to_glomeruli(train_images, pixels)
    test = digits.to_glomeruli(test_images, pixels)
    classifier = FlyClassifier.from_circuit(circuit).fit(train, train_labels)
    predictions = classifier.predict(test)

    print(f"\n--- {len(train)} training digits, one pass, {len(test)} held out ---")
    print(f"  sparsity {classifier.sparsity:.0%} of {circuit.n_kc} Kenyon cells   depression rate {classifier.rate}")
    print(f"  TEST ACCURACY  {accuracy_score(test_labels, predictions):.1%}   (chance would be {1 / classifier.n_classes:.0%})")

    matrix = confusion_matrix(test_labels, predictions)
    report_confusion(matrix)
    report_failures(test_images, test_labels, predictions)
    report_overlap(classifier.encode(train), train_labels, matrix)
    report_sparsity_sweep(circuit, (train, train_labels), (test, test_labels))

    print("\nThe sparsity the fly uses for smells turns out to be about right for pixels too,")
    print("which is the point: the circuit is not specialised for odours, it is a general")
    print("way of turning a dense input into a tag that is easy to learn from.")
    print("\nNext: python 07_draw_a_digit.py")


if __name__ == "__main__":
    main()
