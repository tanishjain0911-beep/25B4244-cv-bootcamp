**Q1. Which class pairs account for most of the noise?**

Noise is dominated by a few **visually similar class pairs**: **truck→automobile (1,247 examples)** and **automobile→truck (1,126)**, followed by **cat→dog (924)** and **dog→cat (834)**. These four confusions account for a substantial share of label errors and show that the noise is **structured around semantic similarity**, not distributed evenly across classes.

**Q2. At which epoch does memorization of wrong labels start, and what is the evidence?**

The evidence places the onset of memorization at **epoch 44**. Training accuracy on mislabeled examples rises from **18.6% at epoch 41** to **21.8% at epoch 44**, then reaches **67.1% by epoch 60**. Meanwhile, noisy-label validation accuracy peaks earlier at **epoch 35 (55.5%)** and declines. The **widening gap between continued training fit and falling validation accuracy** marks the point where additional training increasingly fits wrong labels.

**Q3. Which noise-score signal worked best? Describe one failure mode.**

The **five-fold out-of-fold score 1−P(noisy label)** was the strongest signal, reaching **ROC-AUC 0.8848** and **precision@K 77.09%** at **K equal to the known number of mislabeled training examples**. Its ranking quality is strong, but its main limitation is **ambiguity**: difficult yet correctly labeled images can also receive high noise scores. The score is therefore most useful for **prioritizing review or filtering cautiously**, rather than treating low confidence as proof of a wrong label.

**Q4. Why did your best Task 4 method help, or not help?**

**Co-Teaching was the clear Task 4 winner**, reaching **56.22% noisy-label validation accuracy at epoch 40 (net1)**, compared with **55.10% for label smoothing** and **54.36% for filtering**. Its **1.12-point lead** over label smoothing and **1.86-point lead** over filtering show a consistent practical advantage in these runs. Exchanging each network’s **small-loss examples** is a suitable strategy for this noisy-label setting because it reduces reliance on examples both models find difficult.

**Q5. Did the ranking of methods change between real and synthetic noise? Why?**

Yes—the ranking **reverses**. Co-Teaching, best on **real human-noise labels** in Task 4, scored **56.22%** there, but reached **49.26%** when retrained on **synthetic noise** in Task 5. The Task 1 **cross-entropy baseline scored 49.84%** on synthetic labels, a **0.58-point advantage**. This indicates that Co-Teaching’s benefit is not uniform across noise types; the difference is consistent with its example-selection strategy being more helpful for **structured human-label errors** than for **symmetric random noise**.
