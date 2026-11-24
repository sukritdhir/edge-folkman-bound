import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum

/-!
# The weighted criterion (Lemma 3 of the paper)

This file states and proves, in Lean 4 with Mathlib, the lemma on which the paper
"The edge Folkman number F_e(3,3;4) is at most 483" rests.  It does not check the computations of the
paper; those are done by the programs `verify_q2.py` and `check_independent.py` of the supplement.

## The lemma in the notation of the paper

`G'` is a graph, `I` a set of edges of `G'`, `d` a real number for every edge in `I`, `w` a real number for
every triangle of `G'`, such that the three edges of `T` belong to `I` whenever `w_T ≠ 0`.  `A = A(w,d)` is the
symmetric matrix indexed by `I` with `A_ee = 2 d_e` and `A_ef = 3 w_T` if `e ≠ f` are two edges of a triangle
`T`.  `U` is a set of vertices, `G = G'[U]` and `J = I ∩ E(G)`.  If

* (i) the principal submatrix `A_J` is positive definite, and
* (ii) `∑_{e ∈ J} d_e ≤ 3 ∑_{T ∈ T(G)} w_T` and `J ≠ ∅`,

then every colouring of the edges of `G` with two colours has a monochromatic triangle.

## How this is written here

The graph `G'` of the paper is called `G` here.  Everything is indexed by *ordered* pairs and *ordered*
triples of vertices:

* a vector `y ∈ ℝ^I` is a symmetric function `y : V → V → ℝ` (it is only used on adjacent pairs);
* `d : V → V → ℝ` and `w : V → V → V → ℝ` are arbitrary functions (in the paper they are symmetric);
* `quadForm G w d y` is the sum of `d a b * y a b ^ 2` over the ordered pairs of adjacent vertices plus the
  sum of `w a b c * (y a b * y a c + y a b * y b c + y a c * y b c)` over the ordered triples that form a
  triangle.  Every edge is counted twice and every triangle six times, so for symmetric `d` and `w` this is
  `2 ∑_e d_e y_e² + 6 ∑_T w_T (y_e y_f + y_e y_g + y_f y_g) = yᵀ A y`, the quadratic form of the matrix `A`;
* `InJ G U I a b` says that the pair `a b` is an edge in `J`;
* hypothesis `hw` says that the three edges of a triangle with `w ≠ 0` belong to `I`;
* hypothesis `hpos` is (i): `quadForm G w d y > 0` for every nonzero symmetric `y` that vanishes outside `J`;
* hypothesis `hbudget` is the inequality of (ii) with both sides multiplied by 2:
  `∑ over the ordered pairs in J of d ≤ ∑ over the ordered triangles inside U of w`;
* hypothesis `hne` is `J ≠ ∅`;
* a colouring is a symmetric function `χ : V → V → Bool`; only its values on the edges of `G'[U]` matter, and
  `ArrowsOn G U` says that every such colouring has a triangle inside `U` whose three edges get the same value.
-/

open Finset
open Classical

namespace Folkman

variable {V : Type*} [Fintype V]

/-- The quadratic form `yᵀ A(w,d) y` of the paper, written with ordered pairs and ordered triples. -/
noncomputable def quadForm (G : SimpleGraph V) (w : V → V → V → ℝ) (d : V → V → ℝ) (y : V → V → ℝ) : ℝ :=
  (∑ a, ∑ b, if G.Adj a b then d a b * y a b ^ 2 else 0) +
  ∑ a, ∑ b, ∑ c, if G.Adj a b ∧ G.Adj a c ∧ G.Adj b c then
    w a b c * (y a b * y a c + y a b * y b c + y a c * y b c) else 0

/-- `InJ G U I a b`: the pair `a b` is an edge of the induced subgraph `G[U]` that belongs to `I`. -/
def InJ (G : SimpleGraph V) (U : Set V) (I : V → V → Prop) (a b : V) : Prop :=
  G.Adj a b ∧ a ∈ U ∧ b ∈ U ∧ I a b

/-- Every colouring of the edges of `G[U]` with two colours has a monochromatic triangle. -/
def ArrowsOn (G : SimpleGraph V) (U : Set V) : Prop :=
  ∀ χ : V → V → Bool, (∀ a b, χ a b = χ b a) →
    ∃ a b c, a ∈ U ∧ b ∈ U ∧ c ∈ U ∧ G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ χ a b = χ a c ∧ χ a c = χ b c

/-- **Lemma 3 of the paper (weighted criterion).** -/
theorem weighted_criterion (G : SimpleGraph V) (U : Set V) (I : V → V → Prop)
    (w : V → V → V → ℝ) (d : V → V → ℝ)
    (hI : ∀ a b, I a b → I b a)
    (hw : ∀ a b c, G.Adj a b → G.Adj a c → G.Adj b c → w a b c ≠ 0 → I a b ∧ I a c ∧ I b c)
    (hpos : ∀ y : V → V → ℝ, (∀ a b, y a b = y b a) → (∀ a b, ¬InJ G U I a b → y a b = 0) →
      (∃ a b, y a b ≠ 0) → 0 < quadForm G w d y)
    (hbudget : (∑ a, ∑ b, if InJ G U I a b then d a b else 0) ≤
      ∑ a, ∑ b, ∑ c, if G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ a ∈ U ∧ b ∈ U ∧ c ∈ U then w a b c else 0)
    (hne : ∃ a b, InJ G U I a b) :
    ArrowsOn G U := by
  intro χ hχ
  by_contra hno
  -- the colouring as a vector with entries 1 and -1, extended by 0
  let s : V → V → ℝ := fun a b => if χ a b then 1 else -1
  let y : V → V → ℝ := fun a b => if InJ G U I a b then s a b else 0
  have hs : ∀ a b, s a b = s b a := fun a b => by simp only [s, hχ a b]
  have hs2 : ∀ a b, s a b ^ 2 = 1 := fun a b => by
    simp only [s]; split_ifs <;> norm_num
  have hJsymm : ∀ a b, InJ G U I a b → InJ G U I b a := fun a b h =>
    ⟨h.1.symm, h.2.2.1, h.2.1, hI a b h.2.2.2⟩
  have hy_in : ∀ a b, InJ G U I a b → y a b = s a b := fun a b h => by simp only [y, ite_eq_left h]
  have hy_supp : ∀ a b, ¬InJ G U I a b → y a b = 0 := fun a b h => by simp only [y, ite_eq_right h]
  have hy_symm : ∀ a b, y a b = y b a := by
    intro a b
    by_cases h : InJ G U I a b
    · rw [hy_in a b h, hy_in b a (hJsymm a b h), hs]
    · rw [hy_supp a b h, hy_supp b a (fun h' => h (hJsymm b a h'))]
  have hy_ne : ∃ a b, y a b ≠ 0 := by
    obtain ⟨a, b, h⟩ := hne
    refine ⟨a, b, ?_⟩
    rw [hy_in a b h]
    intro h0
    have := hs2 a b
    rw [h0] at this
    norm_num at this
  have hQ := hpos y hy_symm hy_supp hy_ne
  -- the first sum
  have h1 : (∑ a, ∑ b, if G.Adj a b then d a b * y a b ^ 2 else 0) =
      ∑ a, ∑ b, if InJ G U I a b then d a b else 0 := by
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    by_cases hJ : InJ G U I a b
    · rw [ite_eq_left hJ.1, ite_eq_left hJ, hy_in a b hJ, hs2, mul_one]
    · rw [ite_eq_right hJ, hy_supp a b hJ]
      split_ifs <;> ring
  -- the second sum
  have h2 : (∑ a, ∑ b, ∑ c, if G.Adj a b ∧ G.Adj a c ∧ G.Adj b c then
        w a b c * (y a b * y a c + y a b * y b c + y a c * y b c) else 0) =
      -∑ a, ∑ b, ∑ c,
        if G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ a ∈ U ∧ b ∈ U ∧ c ∈ U then w a b c else 0 := by
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun a _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun c _ => ?_
    by_cases ht : G.Adj a b ∧ G.Adj a c ∧ G.Adj b c
    · rw [ite_eq_left ht]
      by_cases hU : a ∈ U ∧ b ∈ U ∧ c ∈ U
      · -- a triangle of G[U]: it is not monochromatic
        have hc : G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ a ∈ U ∧ b ∈ U ∧ c ∈ U :=
          ⟨ht.1, ht.2.1, ht.2.2, hU.1, hU.2.1, hU.2.2⟩
        rw [ite_eq_left hc]
        by_cases hw0 : w a b c = 0
        · rw [hw0]; ring
        · obtain ⟨iab, iac, ibc⟩ := hw a b c ht.1 ht.2.1 ht.2.2 hw0
          have yab : y a b = s a b := hy_in a b ⟨ht.1, hU.1, hU.2.1, iab⟩
          have yac : y a c = s a c := hy_in a c ⟨ht.2.1, hU.1, hU.2.2, iac⟩
          have ybc : y b c = s b c := hy_in b c ⟨ht.2.2, hU.2.1, hU.2.2, ibc⟩
          have hmono : ¬(χ a b = χ a c ∧ χ a c = χ b c) := fun h =>
            hno ⟨a, b, c, hU.1, hU.2.1, hU.2.2, ht.1, ht.2.1, ht.2.2, h.1, h.2⟩
          have hsum : s a b * s a c + s a b * s b c + s a c * s b c = -1 := by
            simp only [s]
            cases h1 : χ a b <;> cases h2 : χ a c <;> cases h3 : χ b c <;>
              simp_all
          rw [yab, yac, ybc, hsum]; ring
      · -- a vertex of the triangle lies outside U: two of the three values of y vanish
        have hc : ¬(G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ a ∈ U ∧ b ∈ U ∧ c ∈ U) :=
          fun h => hU ⟨h.2.2.2.1, h.2.2.2.2.1, h.2.2.2.2.2⟩
        rw [ite_eq_right hc]
        have hzero : y a b * y a c + y a b * y b c + y a c * y b c = 0 := by
          by_cases ha : a ∈ U
          · by_cases hb : b ∈ U
            · have hcU : c ∉ U := fun h => hU ⟨ha, hb, h⟩
              have e1 : y a c = 0 := hy_supp a c (fun h => hcU h.2.2.1)
              have e2 : y b c = 0 := hy_supp b c (fun h => hcU h.2.2.1)
              rw [e1, e2]; ring
            · have e1 : y a b = 0 := hy_supp a b (fun h => hb h.2.2.1)
              have e2 : y b c = 0 := hy_supp b c (fun h => hb h.2.1)
              rw [e1, e2]; ring
          · have e1 : y a b = 0 := hy_supp a b (fun h => ha h.2.1)
            have e2 : y a c = 0 := hy_supp a c (fun h => ha h.2.1)
            rw [e1, e2]; ring
        rw [hzero]; ring
    · have hc : ¬(G.Adj a b ∧ G.Adj a c ∧ G.Adj b c ∧ a ∈ U ∧ b ∈ U ∧ c ∈ U) :=
        fun h => ht ⟨h.1, h.2.1, h.2.2.1⟩
      rw [ite_eq_right ht, ite_eq_right hc]; ring
  unfold quadForm at hQ
  rw [h1, h2] at hQ
  linarith

end Folkman

-- The axioms on which the proof depends (expected: propext, Classical.choice, Quot.sound).
#print axioms Folkman.weighted_criterion
