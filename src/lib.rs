//! A collection of mate selection methods for evolutionary algorithms

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Mate selection algorithms randomly select pairs of individuals from a population.  
/// The sampling probability of each individuals is a function of its reproductive fitness or "score".  
pub trait MateSelection<R: Rng + ?Sized>: std::fmt::Debug {
    /// Choose multiple weighted pairs
    ///
    /// * Argument `amount` is the number of pairs to return.
    ///
    /// * Argument "scores" is a list containing the reproductive fitness of each individual.
    ///
    /// * Returns a list of pairs of parents to mate together.  
    ///   The parents are specified as indices into the scores list.
    ///
    /// This implementation almost never mates an individual with itself.
    fn pairs(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
        let mut pairs = self.select(rng, amount * 2, scores);

        reduce_repeats(&mut pairs);

        transmute_vec_to_pairs(pairs)
    }

    /// Choose multiple weighted
    fn select(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<usize> {
        if amount == 0 {
            return vec![];
        } else {
            assert!(!scores.is_empty());
        }

        let weights = self.sample_weight(scores);

        stochastic_universal_sampling::choose_multiple_weighted(rng, amount, &weights)
    }

    /// Probability distribution function
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
    /// The sampling weights do **not** need to sum to one.
    fn sample_weight(&self, scores: Vec<f64>) -> Vec<f64>;
}

/// Select parents with a uniform random probability, ignoring the scores.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Random;

/// Select parents with a probability that is directly proportional to their score.
///
/// >   `probability(i) = score(i) / sum(score(x) for x in population)`
///
/// This method biases the selection based on the parents scores. This
/// method is significantly influenced by the magnitude of the fitness
/// scoring function, and by the signal-to-noise ratio between the average
/// score and the variations in the scores.
///
/// Negative or invalid (NaN) scores are discarded and those individuals are
/// not permitted to mate.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Proportional;

/// Normalize the fitness scores into a standard normal distribution.
/// First the scores are normalized into a standard distribution and then they
/// are shifted by the cutoff, which is naturally measured in standard deviations.
/// All scores which are less than the cutoff (now sub-zero) are
/// discarded and those individuals are not permitted to mate.
/// Finally the scores are divided by their sum to yield a selection probability.
/// This method improves upon the proportional method by controlling for the
/// magnitude and variation of the fitness scoring function.
///
/// Argument "**cutoff**" is the minimum negative deviation required for mating.
#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub struct Normalized(pub f64);

/// Select parents from the best ranked individuals in the population.
/// Among the top scoring individuals, individuals are sampled with uniform
/// random probability.
///
/// Argument "**number**" is the number of individuals who are allowed to mate.
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
/// the individuals by their scores in order to rank them from worst to best.
/// The sampling probability is a linear function of the rank.
/// >   `probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))`  
/// >   Where `N` is the population size, and  
/// >   Where `rank = 1` is the best individual and `rank = N` is the worst.  
///
/// Argument "**selection pressure**" measures the inequality in the probability
/// of being selected. Must be in the range [0, 1].
///
/// * At zero, all members are equally likely to be selected.  
/// * At one, the worst ranked individual will never be selected.  
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
    use super::MateSelection;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// A collection of mate selection methods for evolutionary algorithms
    ///
    /// Mate selection algorithms randomly select pairs of individuals from a
    /// population. The sampling probability of each individuals is a function
    /// of its reproductive fitness or "score".
    ///
    /// These implementations almost never mate an individual with itself.
    #[pymodule]
    fn mate_selection(m: Bound<PyModule>) -> PyResult<()> {
        m.add_class::<Random>()?;
        m.add_class::<Proportional>()?;
        m.add_class::<Normalized>()?;
        m.add_class::<Best>()?;
        m.add_class::<Percentile>()?;
        m.add_class::<RankedLinear>()?;
        m.add_class::<RankedExponential>()?;
        Ok(())
    }

    /// Select parents with a uniform random probability, ignoring the scores.
    #[pyclass]
    struct Random(super::Random);

    /// Select parents with a probability that is directly proportional to their score.
    ///
    /// >   probability(i) = score(i) / sum(score(x) for x in population)
    ///
    /// This method biases the selection based on the parents scores. This
    /// method is significantly influenced by the magnitude of the fitness
    /// scoring function, and by the signal-to-noise ratio between the average
    /// score and the variations in the scores.
    ///
    /// Negative or invalid (NaN) scores are discarded and those individuals are
    /// not permitted to mate.
    #[pyclass]
    struct Proportional(super::Proportional);

    /// Normalize the fitness scores into a standard normal distribution. First
    /// the scores are normalized into a standard distribution and then they
    /// are shifted by the cutoff, which is naturally measured in standard
    /// deviations. All scores which are less than the cutoff (now sub-zero)
    /// are discarded and those individuals are not permitted to mate. Finally
    /// the scores are divided by their sum to yield a selection probability.
    /// This method improves upon the proportional method by controlling for
    /// the magnitude and variation of the fitness scoring function.
    ///
    /// Argument "cutoff" is the minimum negative deviation required for mating.
    #[pyclass]
    struct Normalized(super::Normalized);

    /// Select parents from the best ranked individuals in the population.
    /// Among the top scoring individuals, individuals are sampled with uniform
    /// random probability.
    ///
    /// Argument "number" is the number of individuals who are allowed to mate.
    #[pyclass]
    struct Best(super::Best);

    /// Apply a simple percentile based threshold to the population.
    /// Mating pairs are selected with uniform random probability from the
    /// eligible members of the population.
    ///
    /// Argument "percentile" is the fraction of the population which is denied
    /// the chance to mate. At "0" everyone is allowed to mate and at "1" only the
    /// single best individual is allowed to mate.
    #[pyclass]
    struct Percentile(super::Percentile);

    /// Select parents based on their ranking in the population. This method
    /// sorts the individuals by their scores in order to rank them from worst
    /// to best. The sampling probability is a linear function of the rank.
    ///
    /// >   probability(rank) = (1/N) * (1 + SP - 2 * SP * (rank-1)/(N-1))  
    /// >   Where N is the population size, and  
    /// >   Where rank = 1 is the best individual and rank = N is the worst.  
    ///
    /// Argument "selection_pressure" measures the inequality in the
    /// probability of being selected. Must be in the range [0, 1].
    /// * At zero, all members are equally likely to be selected.
    /// * At one, the worst ranked individual will never be selected.
    #[pyclass]
    struct RankedLinear(super::RankedLinear);

    /// Select parents based on their ranking in the population, with an
    /// exponentially weighted bias towards better ranked individuals. This
    /// method can apply more selection pressure than the RankedLinear method
    /// can, which is useful when dealing with very large populations or with a
    /// very large number of offspring.
    ///
    /// Argument "median" describes the exponential slope of the weights curve.
    /// A small median will strongly favor the best individuals, whereas a
    /// large median will sample the individuals more equally. The median is a
    /// rank, and so it is naturally measured in units of individuals.
    /// Approximately half of the sample will be drawn from individuals ranked
    /// better than the median, and the other half will be selected from
    /// individuals with a worse ranking than the median.
    #[pyclass]
    struct RankedExponential(super::RankedExponential);

    #[pymethods]
    impl Random {
        #[new]
        fn new() -> Self {
            Self(super::Random)
        }
        fn __str__(&self) -> String {
            "mate_selection.Random()".to_string()
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Random as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl Proportional {
        #[new]
        fn new() -> Self {
            Self(super::Proportional)
        }
        fn __str__(&self) -> String {
            "mate_selection.Proportional()".to_string()
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Proportional as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl Normalized {
        #[new]
        fn new(cutoff: f64) -> PyResult<Self> {
            if cutoff.is_finite() {
                Ok(Self(super::Normalized(cutoff)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"cutoff\" is not a finite number",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Normalized({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Normalized as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl Best {
        #[new]
        fn new(number: usize) -> PyResult<Self> {
            if number > 0 {
                Ok(Self(super::Best(number)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"number\" is less than one",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Best({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Best as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl Percentile {
        #[new]
        fn new(percentile: f64) -> PyResult<Self> {
            if (0.0..=1.0).contains(&percentile) {
                Ok(Self(super::Percentile(percentile)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"percentile\" is out of bounds [0, 1]",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.Percentile({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::Percentile as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl RankedLinear {
        #[new]
        fn new(selection_pressure: f64) -> PyResult<Self> {
            if (0.0..=1.0).contains(&selection_pressure) {
                Ok(Self(super::RankedLinear(selection_pressure)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"selection_pressure\" is out of bounds [0, 1]",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.RankedLinear({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::RankedLinear as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
        }
    }

    #[pymethods]
    impl RankedExponential {
        #[new]
        fn new(median: usize) -> PyResult<Self> {
            if median > 0 {
                Ok(Self(super::RankedExponential(median)))
            } else {
                Err(PyValueError::new_err(
                    "argument \"median\" is less than one",
                ))
            }
        }
        fn __str__(&self) -> String {
            format!("mate_selection.RankedExponential({})", self.0 .0)
        }
        /// Choose multiple weighted pairs
        /// * Argument "amount" is the number of pairs to return.
        /// * Argument "scores" is the list of reproductive fitness scores.
        /// * Returns a list of pairs of parents to mate together.
        ///   The parents are specified as indices into the scores list.
        fn pairs(&self, amount: usize, scores: Vec<f64>) -> Vec<[usize; 2]> {
            let rng = &mut rand::thread_rng();
            self.0.pairs(rng, amount, scores)
        }
        /// Choose multiple weighted
        fn select(&self, amount: usize, scores: Vec<f64>) -> Vec<usize> {
            let rng = &mut rand::thread_rng();
            self.0.select(rng, amount, scores)
        }
        /// Probability distribution function
        fn pdf(&self, scores: Vec<f64>) -> Vec<f64> {
            <super::RankedExponential as MateSelection<rand::rngs::ThreadRng>>::pdf(&self.0, scores)
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
        stochastic_universal_sampling::choose_multiple(rng, amount, scores.len())
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
        assert!(cutoff.is_finite(), "argument \"cutoff\" is not finite");

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

fn arg_nth_max(amount: usize, data: &[f64]) -> Vec<usize> {
    if amount == 0 {
        return vec![];
    }
    let pivot = data.len() - amount;
    let mut data_copy = data.to_vec();
    let (_, cutoff, _) = data_copy.select_nth_unstable_by(pivot, f64::total_cmp);
    let cutoff = *cutoff;
    let mut index = Vec::with_capacity(amount);
    for (i, x) in data.iter().enumerate() {
        if *x >= cutoff {
            index.push(i)
        }
    }
    // Discard extra elements which are equal to the cutoff.
    if index.len() > amount {
        let mut num_discard = index.len() - amount;
        for cursor in (0..index.len()).rev() {
            if data[index[cursor]] == cutoff {
                index.swap_remove(cursor);
                num_discard -= 1;
                if num_discard == 0 {
                    break;
                }
            }
        }
    }
    index
}

fn zero_and_write_sparse(data: &mut [f64], index: &[usize], value: f64) {
    data.fill(0.0);
    for i in index {
        data[*i] = value;
    }
}

impl Best {
    fn args(&self) -> usize {
        let number = self.0;
        assert!(number > 0, "argument \"number\" is less than one");
        number
    }
}
impl<R: Rng + ?Sized> MateSelection<R> for Best {
    fn select(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<usize> {
        let num_best = self.args();
        let index = arg_nth_max(num_best, &scores);
        let sample = stochastic_universal_sampling::choose_multiple(rng, amount, index.len());
        sample.iter().map(|&s| index[s]).collect()
    }
    fn pdf(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let num_best = self.args();
        let index = arg_nth_max(num_best, &scores);
        zero_and_write_sparse(&mut scores, &index, 1.0 / num_best as f64);
        scores
    }
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let num_best = self.args();
        let index = arg_nth_max(num_best, &scores);
        zero_and_write_sparse(&mut scores, &index, 1.0);
        scores
    }
}

impl Percentile {
    fn get_index(&self, scores: &[f64]) -> Vec<usize> {
        let percentile = self.0;
        assert!(
            (0.0..=1.0).contains(&percentile),
            "argument \"percentile\" is out of bounds [0, 1]"
        );
        let num_eligible = ((1.0 - percentile) * scores.len() as f64).round() as usize;
        arg_nth_max(num_eligible.max(1), &scores)
    }
}
impl<R: Rng + ?Sized> MateSelection<R> for Percentile {
    fn select(&self, rng: &mut R, amount: usize, scores: Vec<f64>) -> Vec<usize> {
        let index = self.get_index(&scores);
        let sample = stochastic_universal_sampling::choose_multiple(rng, amount, index.len());
        sample.iter().map(|&s| index[s]).collect()
    }
    fn pdf(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let index = self.get_index(&scores);
        zero_and_write_sparse(&mut scores, &index, 1.0 / index.len() as f64);
        scores
    }
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let index = self.get_index(&scores);
        zero_and_write_sparse(&mut scores, &index, 1.0);
        scores
    }
}

impl<R: Rng + ?Sized> MateSelection<R> for RankedLinear {
    fn sample_weight(&self, mut scores: Vec<f64>) -> Vec<f64> {
        let selection_pressure = self.0;
        assert!(
            (0.0..=1.0).contains(&selection_pressure),
            "argument \"selection_pressure\" is out of bounds [0, 1]"
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
        assert!(median > 0, "argument \"median\" is less than one");
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
        let pairs = Proportional.pairs(rng, 0, vec![]);
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
    fn all_equal_to_the_best() {
        let rng = &mut rand::thread_rng();
        Best(3).select(rng, 1, vec![4.0, 4.0, 4.0, 4.0]);
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
