//! This module is a collection of mate-selection strategies for evolutionary algorithms.

use rand::prelude::{Rng, SliceRandom};
use serde::{Deserialize, Serialize};

/// Mate selection algorithms randomly select pairs of individuals from a
/// population, with the intent of mating them together. Individual are
/// selected based on their reproductive fitness "score".
pub trait MateSelection<R: Rng + ?Sized>: std::fmt::Debug {
    /// Apply the mate selection algorithm.
    ///
    /// Argument `amount` is the desired number of mating pairs.
    /// This almost never mates an individual with itself.
    ///
    /// Argument `scores` contains the reproductive fitness of each individual.
    ///
    /// Returns a list of pairs of parents to mate together. The parents are
    /// specified as indices into the scores list.
    fn pairs(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
        let mut pairs = self.select(rng, amount * 2, scores);

        reduce_repeats(&mut pairs);

        transmute_vec_to_pairs(pairs)
    }

    /// Choose multiple weighted
    fn select(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<usize> {
        if amount == 0 || scores.is_empty() {
            return vec![];
        }

        let weights = self.sample_weight(scores);

        stochastic_universal_sampling::choose_multiple_weighted(rng, amount, &weights)
    }

    /// Probability Density Function
    fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
        let mut pdf = self.sample_weight(scores);
        // Normalize the sum to one.
        let sum: f64 = pdf.iter().sum();
        let div_sum = 1.0 / sum;
        for x in pdf.iter_mut() {
            *x *= div_sum;
        }
        pdf
    }

    /// Transform the reproductive fitness scores into sampling weights.  
    /// Each implementation of this trait has a different algorithm here.  
    fn sample_weight(&self, scores: Vec<f64>) -> Vec<f64>;
}

/// Select parents with a uniform random probability, ignoring the scores.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Random;

/// Select parents with a probability that is directly proportional to the
/// magnitude of their score.
/// >   `probability(i) = score(i) / sum(score(x) for x in population)`
///
/// Typically this method does not directly prevent any individuals from
/// mating, instead it biases the selection based on their scores. This
/// method is significantly influenced by the magnitude of the fitness
/// scoring function, and by the signal-to-noise ratio between the average
/// score and the variations in the scores.
///
/// Negative or invalid (NaN) scores are discarded and those individuals are
/// not permitted to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Proportional;

/// Normalize the fitness scores into a standard normal distribution.
/// First the scores are normalized into a standard distribution and then
/// shifted by the cutoff, which is naturally measured in standard
/// deviations. All scores which are less than the cutoff (now sub-zero)
/// are discarded and those individuals are not permitted to mate. Finally
/// the scores are divided by their sum to yield a selection probability.
/// This method improves upon the proportional method by controlling for the
/// magnitude and variation of the fitness scoring function.
///
/// Argument "**cutoff**" is the minimum negative deviation required for mating.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Normalized(pub f64);

/// Select parents from the best ranked individuals in the population.
///
/// Argument "**amount**" is the number of individuals who are allowed to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
struct Best(pub usize);

/// Apply a simple percentile based threshold to the population.
/// Mating pairs are selected with uniform random probability from the eligible
/// members of the population.
///
/// Argument "**percentile**" is the fraction of the population which is denied
/// the chance to mate. At `0` everyone is allowed to mate and at `1` only the
/// single best individual is allowed to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Percentile(pub f64);

/// Select parents based on their ranking in the population. This method sorts
/// the individuals by their scores. Statistically, better ranked individuals
/// will have more children than worse ranked individuals.
/// >   `probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))`  
/// >   Where `N` is the population size, and  
/// >   Where `rank = 1` is the best individual and `rank = N` is the worst.  
///
/// Argument "**selection pressure**" measures the inequality of the probability
/// of being selected. Must be in the range [0, 1].
///
/// At zero, all members are equally likely to be selected.  
/// At one, the worst ranked individual will never be selected.  
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct RankedLinear(pub f64);

/// Select parents based on their ranking in the population, with an
/// exponentially weighted bias towards better ranked individuals. This method
/// can apply more selection pressure than the RankedLinear method can, which
/// is useful when dealing with very large populations or with a very large
/// number of offspring.
///
/// Argument "**median**" describes the exponential slope of the weights curve.
/// A small median will strongly favor the best individuals, whereas a
/// large median will sample the individuals more equally. The median is a
/// rank, and so it is naturally measured in units of individuals.
/// Approximately half of the sample will be drawn from individuals ranked
/// better than the median, and the other half will be selected from
/// individuals with a worse ranking than the median.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct RankedExponential(pub usize);

#[cfg(feature = "pyo3")]
mod python {
    use pyo3::prelude::*;
    use serde::{Deserialize, Serialize};

    #[pymodule]
    fn mate_selection(m: Bound<PyModule>) -> PyResult<()> {
        m.add_class::<MateSelection>()?;
        Ok(())
    }

    /// This algorithm determines which individuals will mate with each other.
    #[pyclass]
    #[derive(Serialize, Deserialize)]
    struct MateSelection(super::MateSelection);

    #[pymethods]
    #[allow(non_snake_case)]
    impl MateSelection {
        // TODO: All of these doc strings need to make clear that they are constructor methods.

        /// Select parents with a uniform random probability.
        #[staticmethod]
        fn Random() -> Self {
            Self(super::MateSelection::Random)
        }

        /// TODO
        #[staticmethod]
        fn Best(amount: usize) -> Self {
            Self(super::MateSelection::Best(amount))
        }

        /// Allow only the higher scoring fraction of the population to mate.
        /// Mating pairs are selected with uniform random probability from the
        /// eligible members of the population. This method does not bias the
        /// selection based on the score, beyond a simple percentile-based threshold.
        #[staticmethod]
        fn Percentile(percentile: f64) -> Self {
            assert!((0.0..=1.0).contains(&percentile));
            Self(super::MateSelection::Percentile(percentile))
        }

        /// Select parents with a probability that is directly proportional to the
        /// magnitude of their score.
        /// >   probability(i) = score(i) / sum(score(x) for x in population)
        ///
        /// Typically this method does not directly prevent any individuals from
        /// mating, instead it biases the selection based on their scores. This
        /// method is significantly influenced by the magnitude of the fitness
        /// scoring function, and by the signal-to-noise ratio between the average
        /// score and the variations in the scores.
        ///
        /// Negative or invalid (NaN) scores are discarded and those individuals are
        /// not permitted to mate.
        #[staticmethod]
        fn Proportional() -> Self {
            Self(super::MateSelection::Proportional)
        }

        /// Normalize the fitness scores into a standard normal distribution.
        /// First the scores are normalized into a standard distribution and
        /// then shifted by the cutoff, which is naturally measured in standard
        /// deviations. All scores which are less than the cutoff (now sub-zero)
        /// are discarded and those individuals are not permitted to mate. Finally
        /// the scores are divided by their sum to yield a selection probability.
        /// This method improves upon the proportional method by controlling for the
        /// magnitude and variation of the fitness scoring function.
        #[staticmethod]
        fn Normalized(cutoff: f64) -> Self {
            assert!(cutoff.is_finite());
            Self(super::MateSelection::Normalized(cutoff))
        }

        /// Select parents based on the relative rankings between them within the
        /// population. Statistically, better ranked individuals will have more
        /// children than worse ranked individuals. This method does not consider
        /// the actual magnitude of the fitness scores, it only uses the scores
        /// to order and rank the members of the population relative to each other.  
        /// >   probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))
        /// >   Where N is the population size, and
        /// >   Where rank = 1 is the best individual and rank = N is the worst.
        ///
        /// Argument selection_pressure measures the inequality of the
        /// probability of being selected. Must be in the range [0, 1].
        ///
        /// At zero, all members are equally likely to be selected.
        /// At one, the worst ranked individual will never be selected.
        #[staticmethod]
        fn RankedLinear(selection_pressure: f64) -> Self {
            assert!((0.0..=1.0).contains(&selection_pressure));
            Self(super::MateSelection::RankedLinear(selection_pressure))
        }

        /// Select parents based on the relative rankings between them, with an
        /// exponentially weighted bias towards better ranked individuals. This
        /// method does not consider the actual magnitude of the fitness scores,
        /// it only uses the scores to order and rank the members of the population
        /// relative to each other. This method can apply more selection pressure
        /// than the RankedLinear method can, which is useful when dealing with very
        /// large populations or with a very large number of offspring.
        ///
        /// Argument median describes the exponential slope of the weights curve.
        /// A small median will strongly favor the best individuals, whereas a
        /// large median will sample the individuals more equally. The median
        /// is a rank, and so it is naturally measured in units of individuals.
        /// Approximately half of the sample will be drawn from individuals
        /// ranked better than the median, and the other half will be selected
        /// from individuals with a worse ranking than the median.
        #[staticmethod]
        fn RankedExponential(median: usize) -> Self {
            assert!(median > 0);
            Self(super::MateSelection::RankedExponential(median))
        }

        /// Probability Density Function.
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            self.0.pdf(scores)
        }

        /// TODO
        fn select(&self, scores: Vec<f64>, amount: usize) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, scores, amount)
        }

        /// Apply the selection algorithm.
        ///
        /// Argument amount is the number of mating pairs to return. Note that on
        /// occasion a parent may be mated with itself.
        ///
        /// Argument scores contains the reproductive fitness of each individual.
        ///
        /// Returns a list of pairs of parents to mate together. The parents are
        /// specified as indexes into the scores list.
        fn pairs(&self, scores: Vec<f64>, amount: usize) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, scores, amount)
        }

        fn __str__(&self) -> String {
            format!("{:?}", self.0)
        }

        fn __repr__(&self) -> String {
            format!("<{:?}>", self.0)
        }

        fn __eq__(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for Random {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        scores.fill(1.0);
        scores
    }

    fn pdf(&self, mut scores: Vec<f64>) -> Vec<f64> {
        if !scores.is_empty() {
            let p = 1.0 / scores.len() as f64;
            scores.fill(p);
        }
        scores
    }

    fn select(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<usize> {
        if amount == 0 || scores.is_empty() {
            return vec![];
        }
        //
        let mut indices = Vec::<usize>::with_capacity(amount);
        while indices.len() < amount {
            let remaining = amount - indices.len();
            if remaining >= scores.len() {
                indices.extend(0..scores.len());
            } else {
                indices.extend(rand::seq::index::sample(rng, scores.len(), remaining));
            }
        }
        indices.shuffle(rng);
        indices
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for Best {
    fn sample_weight(&self, _scores: Vec<f64>) -> Vec<f64> {
        let amount = self.0;
        assert!(amount != 0, "argument out of bounds");

        todo!()
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for Percentile {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let percentile = self.0;
        assert!((0.0..=1.0).contains(&percentile), "argument out of bounds");

        let cutoff = (percentile * scores.len() as f64).round() as usize;
        let cutoff = cutoff.min(scores.len() - 1);
        let mut scores_copy = scores.to_vec();
        let (_, cutoff, _) = scores_copy.select_nth_unstable_by(cutoff, f64::total_cmp);
        let cutoff = *cutoff;
        // Apply the truncation cutoff to the scores vector, yielding
        // weights of either 0.0 or 1.0.
        for x in scores.iter_mut() {
            *x = f64::from(*x >= cutoff);
        }
        scores
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for Proportional {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        // Replace negative & invalid values with zero.
        for x in scores.iter_mut() {
            *x = x.max(0.0);
        }
        scores
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for Normalized {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let cutoff = self.0;
        assert!(cutoff.is_finite(), "argument is not finite");

        // Find and normalize by the average score.
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        for x in scores.iter_mut() {
            *x -= mean;
        }
        // Find and normalize by the standard deviation of the scores.
        let var = scores.iter().map(|x| x.powi(2)).sum::<f64>() / scores.len() as f64;
        let std = var.sqrt();
        for x in scores.iter_mut() {
            // Shift the entire distribution and cutoff all scores which
            // are less than zero.
            *x = (*x / std - cutoff).max(0.0);
        }
        scores
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for RankedLinear {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let selection_pressure = self.0;
        assert!(
            (0.0..=1.0).contains(&selection_pressure),
            "argument out of bounds"
        );

        let div_n = if scores.len() == 1 {
            0.0 // Value does not matter, just don't crash.
        } else {
            1.0 / (scores.len() - 1) as f64
        };
        for (rank, index) in argsort(&scores).iter().enumerate() {
            // Reverse the ranking from ascending to descending order
            // so that rank 0 is the best & rank N-1 is the worst.
            let rank = scores.len() - 1 - rank;
            // Scale the ranking into the range [0, 1].
            let rank = rank as f64 * div_n;
            scores[*index] = 1.0 + selection_pressure - 2.0 * selection_pressure * rank;
        }
        scores
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for RankedExponential {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let median = self.0;
        assert!(median != 0, "argument out of bounds");
        for (rank, index) in argsort(&scores).iter().enumerate() {
            let rank = scores.len() - rank - 1;
            scores[*index] = (-(2.0_f64.ln()) * rank as f64 / median as f64).exp();
        }
        scores
    }
}

fn argsort(scores: &[f64]) -> Vec<usize> {
    let mut argsort: Vec<_> = (0..scores.len()).collect();
    argsort.sort_unstable_by(|a, b| f64::total_cmp(&scores[*a], &scores[*b]));
    argsort
}

/// This helps avoid mating an individual with itself.
fn reduce_repeats(data: &mut [usize]) {
    debug_assert!(is_even(data.len()));
    // Simple quadratic greedy algorithm for breaking up pairs of repeated elements.
    // First search for pairs of repeated values.
    'outer: for cursor in (0..data.len()).step_by(2) {
        let value = data[cursor];
        if value == data[cursor + 1] {
            // Then find a different value to swap with.
            for search in (cursor + 2..data.len()).step_by(2) {
                if data[search] != value && data[search + 1] != value {
                    data.swap(cursor, search);
                    continue 'outer;
                }
            }
            for search in (0..cursor).step_by(2) {
                if data[search] != value && data[search + 1] != value {
                    data.swap(cursor, search);
                    continue 'outer;
                }
            }
        }
    }
}

/// Transmute the vector of samples into pairs of samples, without needlessly copying the data.
fn transmute_vec_to_pairs(data: Vec<usize>) -> Vec<[usize; 2]> {
    // Check that there are an even number of values in the vector.
    assert!(is_even(data.len()));
    // Check the data alignment.
    assert_eq!(
        std::mem::align_of::<usize>(),
        std::mem::align_of::<[usize; 2]>()
    );
    // Take manual control over the data vector.
    let mut data = std::mem::ManuallyDrop::new(data);
    unsafe {
        // Disassemble the vector.
        let ptr = data.as_mut_ptr();
        let mut len = data.len();
        let mut cap = data.capacity();
        // Transmute the vector.
        let ptr = std::mem::transmute::<*mut usize, *mut [usize; 2]>(ptr);
        len /= 2;
        cap /= 2;
        // Reassemble and return the data.
        Vec::from_raw_parts(ptr, len, cap)
    }
}

const fn is_even(x: usize) -> bool {
    x & 1 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flatten_and_sort(pairs: &Vec<[usize; 2]>) -> Vec<usize> {
        let mut data: Vec<usize> = pairs.iter().flatten().copied().collect();
        data.sort_unstable();
        data
    }

    #[test]
    fn is_even() {
        assert!(super::is_even(0));
        assert!(!super::is_even(1));
        assert!(super::is_even(2));
        assert!(!super::is_even(3));
    }

    #[test]
    fn no_data() {
        let rng = &mut rand::thread_rng();
        let pairs = Proportional.pairs(rng, 1, vec![]);
        assert!(pairs.is_empty());

        let pairs = Proportional.pairs(rng, 0, vec![1.0, 2.0, 3.0]);
        assert!(pairs.is_empty());
    }

    #[test]
    fn truncate_top_one() {
        let rng = &mut rand::thread_rng();
        // Truncate all but the single best individual.
        let algo = Percentile(0.99);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(rng, 1, weights);
        assert!(pairs == [[99, 99]]);
    }

    #[test]
    fn truncate_top_two() {
        let rng = &mut rand::thread_rng();
        // Truncate all but the best two individuals.
        let algo = Percentile(0.98);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(rng, 1, weights);
        assert!(pairs == [[98, 99]] || pairs == [[99, 98]]);
    }

    #[test]
    fn truncate_none() {
        let rng = &mut rand::thread_rng();
        // Truncate none of the individuals.
        let algo = Percentile(0.0);
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(rng, 50, weights);
        let selected = flatten_and_sort(&pairs);
        assert_eq!(selected, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn truncate_all() {
        let rng = &mut rand::thread_rng();
        // Truncating all individuals should actually just return the single
        // best individual. This situation happens when building the starting
        // population.
        let algo = Percentile(0.999_999_999); // Technically less than one.
        let weights: Vec<f64> = (0..100).map(|x| x as f64 / 100.0).collect();
        let pairs = algo.pairs(rng, 1, weights);
        assert!(pairs == [[99, 99]]);
    }

    #[test]
    fn propotional() {
        let rng = &mut rand::thread_rng();
        // All scores are equal, proportional should select all of the items.
        let weights = vec![1.0; 10];
        let algo = Proportional;
        let selected = flatten_and_sort(&algo.pairs(rng, 5, weights));
        assert_eq!(selected, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn propotional_outlier() {
        let rng = &mut rand::thread_rng();
        // Index 0 is an outlier. Proportional selection should allow the
        // outlier to dominate the sample. The other items should not be selected.
        let weights = vec![1000_000_000_000_000.0, 1.0, 1.0, 1.0];
        let algo = Proportional;
        let selected = flatten_and_sort(&algo.pairs(rng, 10, weights));
        let inliers: Vec<_> = selected.iter().filter(|&idx| *idx != 0).collect();
        assert!(inliers.is_empty());
    }

    #[test]
    fn propotional_negative() {
        let rng = &mut rand::thread_rng();
        // One score is extremely negative and another is NAN.
        // Proportional should ignore them.
        let mut weights = vec![1.0; 12];
        weights[5] = -100.0;
        weights[6] = f64::NAN;
        let algo = Proportional;
        let selected = flatten_and_sort(&algo.pairs(rng, 5, weights));
        assert_eq!(selected, [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn normalized() {
        let rng = &mut rand::thread_rng();
        // Normalize can deal with negative scores, it does not care about their absolute values.
        let weights = vec![-20.0, -12.0, -11.0, -10.5, -10.0, -9.5, -9.0, -8.0, 0.0];
        const MEAN_IDX: usize = 4;
        const MAX_IDX: usize = 8;
        let cutoff = -0.01;
        let algo = Normalized(cutoff);
        let selected = flatten_and_sort(&algo.pairs(rng, 2, weights));
        // Only scores greater than the mean should have been selected.
        assert!(selected.iter().all(|&x| x >= MEAN_IDX));
        // The sample should contain the highest score, but not be dominated by it.
        assert!(selected.contains(&MAX_IDX));
        assert!(!selected.iter().all(|&x| x == MAX_IDX));
    }

    #[test]
    fn ranked_linear() {
        let rng = &mut rand::thread_rng();
        // Index 0 is an outlier.
        // Ranking the scores should prevent the outlier from dominating.
        let weights = vec![1000_000_000_000_000.0, 1.0, 1.0, 1.0];

        // No selection pressure, should select all four scores.
        let algo = RankedLinear(0.0);
        let selected = flatten_and_sort(&algo.pairs(rng, 2, weights));
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    /// Finds those off-by-one errors.
    #[test]
    fn ranked_linear_single() {
        let rng = &mut rand::thread_rng();
        let weights = vec![4.0];
        let algo = RankedLinear(0.5);
        let selected = flatten_and_sort(&algo.pairs(rng, 1, weights));
        assert_eq!(selected, vec![0, 0]);
    }

    #[test]
    fn ranked_linear_outlier() {
        let rng = &mut rand::thread_rng();
        // Index 0 is an outlier.
        // Ranking the scores should prevent the outlier from dominating.
        let mut weights = vec![1000_000_000_000_000.0];
        weights.append(&mut vec![1.0; 1000]);
        // With selection pressure, the outlier still should not dominate the sampling.
        let algo = RankedLinear(1.0);
        let selected = flatten_and_sort(&algo.pairs(rng, 10, weights));
        let inliers: Vec<_> = selected.iter().filter(|&idx| *idx != 0).collect();
        assert!(!inliers.is_empty());
    }

    #[test]
    fn ranked_exponential() {
        let rng = &mut rand::thread_rng();
        let test_cases = [
            (1, 1, 2, 99), // Test selecting with one single weight does not crash.
            (3, 1, 4, 1),
            (100, 10, 100, 5),
            (1000, 10, 100, 5),
            (10_000, 100, 10_000, 20),
            (10_000, 1000, 10_000, 50),
        ];
        for (num, median, sample, tolerance) in test_cases {
            let weights: Vec<f64> = (0..num).map(|x| x as f64).collect();
            let algo = RankedExponential(median);
            assert_eq!(sample, (sample / 2) * 2); // Sample count needs to be even for this to work.
            let selected = flatten_and_sort(&algo.pairs(rng, sample / 2, weights));
            dbg!(&selected);
            // Count how many elements are from the top ranked individuals.
            let top_count_actual = selected
                .iter()
                .filter(|&&idx| idx >= (num - median))
                .count();
            let top_count_desired = sample / 2;
            dbg!(num, median, sample, tolerance);
            dbg!(top_count_actual, top_count_desired);
            assert!((top_count_actual as i64 - top_count_desired as i64).abs() <= tolerance);
            assert!(top_count_actual > 0);
        }
    }

    /// Check that this avoids mating individuals with themselves.
    #[test]
    fn pairs() {
        let rng = &mut rand::thread_rng();
        // N is the population size.
        // P is the number of mating pairs.
        // R is the percent of the pairs that are duplicates.
        for (n, max_r) in [
            (2, 5.0),
            (3, 4.0),
            (4, 3.0),
            (5, 3.0),
            (10, 3.0),
            (20, 2.0),
            (100, 1.0),
            //
        ] {
            let p = 10 * n;
            // let p = 3;
            let indices = Random.pairs(rng, p, vec![1.0; n]);
            let num_repeats = indices.iter().filter(|[a, b]| a == b).count();
            let percent_repeats = 100.0 * num_repeats as f64 / indices.len() as f64;

            println!("Population Size = {n}, Mating Pairs = {p}, Repeats = {percent_repeats:.2} %");
            dbg!(indices);
            assert!(percent_repeats <= max_r);
        }
    }

    /// Example of the trait used as an argument.
    #[test]
    fn argument() {
        type Rng = rand::rngs::ThreadRng;

        fn foobar(select: &dyn MateSelection<Rng>) {
            let rng = &mut rand::thread_rng();
            select.select(rng, 0, vec![]);
        }

        let x: &dyn MateSelection<Rng> = if rand::random() {
            &Random
        } else {
            &Proportional
        };

        foobar(x);
    }
}
