# test_summarizer.py
from summarizers.generative_summarizer import GenerativeSummarizer

# Texte à résumer
long_text = """
royalsocietypublishing.org/journal/rsta
Review
Article submitted to journal
Subject Areas:
machine learning, astrophysics
Keywords:
symbolic regression, minimum
description length
Author for correspondence:
Harry Desmond
e-mail: harry.desmond@port.ac.uk
(Exhaustive) Symbolic
Regression and model
selection by minimum
description length
Harry Desmond1
1
Institute of Cosmology & Gravitation, University of
Portsmouth, Dennis Sciama Building,
Portsmouth, PO1 3FX, United Kingdom
Symbolic regression is the machine learning method
for learning functions from data. After a brief
overview of the symbolic regression landscape, I will
describe the two main challenges that traditional
algorithms face: they have an unknown (and likely
significant) probability of failing to find any given
good function, and they suffer from ambiguity
and poorly-justified assumptions in their functionselection procedure. To address these I propose
an exhaustive search and model selection by the
minimum description length principle, which allows
accuracy and complexity to be directly traded off by
measuring each in units of information. I showcase
the resulting publicly available Exhaustive Symbolic
Regression algorithm on three open problems in
astrophysics: the expansion history of the universe,
the effective behaviour of gravity in galaxies and
the potential of the inflaton field. In each case
the algorithm identifies many functions superior
to the literature standards. This general purpose
methodology should find widespread utility in
science and beyond.
1. Introduction
A key activity in science is summarising observational
data with functional fits. Either one wants a “fitting
function” with which to propagate some correlation into
another part of the analysis, or one wishes to learn
the “law” that governs the data. Traditionally these
two tasks have been tackled in different ways. The
creation of fitting functions is normally done “by eye”:
one plots the data and estimates the types of operators
and their composition that may give a good fit. This is
supplemented with a trial-and-error step: if a given func-
© The Authors. Published by the Royal Society under the terms of the
Creative Commons Attribution License http://creativecommons.org/licenses/
by/4.0/, which permits unrestricted use, provided the original author and
source are credited.
arXiv:2507.13033v1 [astro-ph.IM] 17 Jul 2025
2
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
tional form does not give quite the right asymptotic behaviour, for example, another one is tried
and the process iterated until a satisfactory fit is achieved. Such a procedure has on occasion been
implemented also for learning “scientific laws”, the most notable example being Planck’s (and his
antecedents’) discovery of (parts of) the blackbody function. More often however, the discovery
of laws is achieved primarily by theory. Partially or completely independently of data, the theorist
proposes principles or hypotheses that lead to certain functional relations between observables.
These functions are then tested on the data to assess the theory’s veracity.
This traditional approach has drawbacks. How can one be sure that the fitting functions one
creates are in any sense optimal? One way is for the function to find a theoretical underpinning, as
happened with the blackbody formula through quantum mechanics. But purely empirically there
is no guarantee that any particular aspect of the function that may be important in applications
of it—say its interpolation or extrapolation behaviour—is robust unless one has a quantitative
assessment of the function’s quality relative to others. Even were such a quality metric available,
assessing (all?) other possible functions seems infeasible. The same concerns apply to physical
laws extracted empirically: the true features of the law may not be captured by an imperfect
function generation procedure. The top-down approach (creating functions in an extra-empirical
way before bringing in the data) of course has different concerns: in that case one must assess the
reliability of theoretical arguments rather than a regression procedure. Might there be a way to
greatly expand the capacity to learn laws directly from data, mitigating or even obviating these
concerns?
Enter symbolic regression (SR), the machine learning (ML) framework for extracting functions
directly from data. SR aims to automate and perfect the process described above. To explain it,
it is useful to begin with the better-known form of regression, which I term regular or numerical.
Here one specifies a priori the functional form that one wants to fit and the regression is only
over the numerical values of the function’s free parameters. SR generalises this procedure by
bringing the functional form itself—the operators and their ordering—into the search space. This
substantially increases the difficulty: not only is the search space much enlarged, but the lack of
continuity between operators invalidates the use of techniques such as gradient descent that form
the staple of parameter optimisation. But there are also clear advantages, most obviously that in
regular regression the functional form one imposes is likely to be suboptimal or just plain wrong.
SR removes confirmation bias because the user is not required to make that important decision.
While regular regression has a long and venerable history, the advent of ML has introduced
some additional “competitors” to SR. These are methods like neural networks, Gaussian processes
and random forests which can capture the correlations present in a dataset to high fidelity by
compounding very many simple functions. These methods excel at “mindless” regression and
classification tasks: they can produce accurate predictions within the domain of their training
set, but very rarely produce scientific insight into a system. This makes them best suited to
summarising correlations that one wishes to treat entirely as “nuisance”, either because the
underlying physics is already thoroughly well-known (e.g. the emergent behaviour of “baryonic
physics” in galaxy evolution), or because it is not believed to be possible to learn basic science
from them. SR, on the other hand, excels in cases where one does care about the functional
form of a relation, perhaps because one believes it to reflect the meaningful physics governing
the system. When successful, SR uncovers either the true equation that generated the data—
if there is one—or else simply the best possible functional representation. That said, SR can
also be valuable for constructing emulators without any demand for, or benefit to be gained
from, interpretability: such symbolic emulators are highly portable (e.g. they do not require the
neural network emulator’s constrained weights and biases), rapid to evaluate and potentially
very accurate (e.g. [1]).
Exploring the parameter space is quite different when this includes operators. The most
common method for generating trial functions is called a genetic algorithm (GA), which works
by analogy with natural selection [2–4]. This begins by creating a population of functions with
typically random examples. One then calculates the “fitness” of each one, which is its accuracy on
3
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
the target dataset, and kills off the functions that fail some fitness cut. Then the next generation of
functions is produced through mutation and crossover. A mutation changes one parts of a function,
either a single operator or connected set of operators. A crossover mixes parts of two functions by
swapping branches to produce two new functions. This new generation is then assessed against a
stopping criterion—for example whether a function of sufficient accuracy has been produced, or
a time limit—and if the criterion is not met the process is repeated in the hope of producing everbetter functions. Popular algorithms in this category are Operon [5], PySR [6] and DataModeler
(a proprietary Mathematica add-on; [7]).
This review focuses on an alternative, non-stochastic approach to SR. A particular emphasis
will be the model selection metric, i.e. the quantity used to determine how good trial functions are.
I will begin by describing the traditional way in which functions are evaluated (Sec. 2). I will then
describe the approach of Exhaustive Symbolic Regression, which overhauls both the function
generation and function assessment methodology (Sec. 3). Emphasis here is on the minimum
description length selection metric (Sec. 4). I will then describe three astrophysical applications
of ESR+MDL, showcasing its power on the expansion rate of the universe (Sec. 5(a)), galaxy
dynamics (Sec. 5(b)) and cosmic inflation (Sec. 5(c)). I then describe future developments and
conclude. Throughout the article log has base e.
2. Traditional function assessment
Suppose we have a set of trial functions (e.g. from a GA)—how should we score them? Let us
consider again the analogy with regular regression. The analogues of candidate functions in this
case are points in the space of the pre-defined function’s free parameters. In a Bayesian context
these are scored by two metrics: the likelihood they give the data, and their prior probability.
These multiply by Bayes’ theorem, normalised by the evidence, to form the posterior. The best
solution is the one that maximises the posterior, with an uncertainty given by a confidence interval
of the posterior probability distribution. In a frequentist context one would use just the likelihood.
In SR, focusing on the likelihood or posterior produces an immediate problem. Now that we
are allowed to vary operators we can produce arbitrarily complex functions, which typically
means that they can be made arbitrarily accurate on the dataset in question. As an example,
consider that one can perfectly fit any set of N datapoints with a polynomial of degree N − 1. But
such extremely complex functions that maximise the likelihood are likely severely overfitted and
hence extrapolate or generalise very poorly. This means that some measure of simplicity needs
to be included in the function selection procedure: one wants some optimal trade-off between
simplicity and accuracy.
The simplest way of doing this is (unfortunately) the approach traditionally taken in SR;
this is to include simplicity as an incommensurate second objective in the regression. This is
quantified with a “complexity” heuristic, for example the number of nodes in the function’s tree
representation (i.e. the number of operators, parameters or variables in the function). Functions
are then plotted on the 2D plane of accuracy (measured by maximum likelihood or posterior
value, or its poor-man’s version mean square error) and complexity, producing a Pareto (twoobjective) optimisation problem. There is a privileged set of functions on the Pareto plane, which
are the most accurate for their complexity. These functions—one at each complexity value—form
the Pareto front and are termed “Pareto optimal”: any other function has higher inaccuracy and/or
complexity and is “Pareto dominated” by the optimum functions.
From this perspective all functions along the Pareto front are “the best”, and they cannot be
compared. No metric is provided for how much gain in accuracy is required for an increment of
complexity to be warranted. A second heuristic is therefore required for deciding where along
the Pareto front to select the best function(s) from. One could eyeball the functions and pick the
one deemed to be most attractive, one could decide one wants a function of a certain complexity,
one could score the functions on both a training and test dataset and look for the complexity
at which the accuracy of the latter starts to become significantly worse than that of the former
(indicative of overfitting), or one could stipulate some function of accuracy and complexity to
4
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
determine the final score. These procedures have no rigorous justification and are up to the user,
yet may radically alter the outcome of the regression. We will see in the next section that a superior
method exists.
3. Exhaustive Symbolic Regression
(a) Motivation and Operation
The traditional approach to SR outlined above faces two major challenges. First, a stochastic
search has some completely unknown probability of failing to find any given good function. This
may not be a serious issue if that probability is small in practice, but we will see in Sec. 3(b) that
this is not the case. The failure probability is a function of the algorithm’s hyperparameters, but
one does not know in advance how to set those to maximise efficacy for the dataset in question.
This makes SR results untrustworthy, as there may be any number of better-performing functions
than those found. Second, judging equations on their Pareto-optimality depends crucially on
the definitions of accuracy and complexity. Changing these will move functions around on the
accuracy–complexity plane and change the ones that lie on the Pareto front. Most algorithms
adopt mean square error (MSE) as the accuracy measure, but this only accurately describes
the data likelihood if the uncertainties on the data points are Gaussian and constant. More
importantly the complexity definition is largely arbitrary: some approaches use the number of
operators, parameters and variables in the function, some use the depth of the function’s tree
representation, while others adopt behavioural rather than structural measures like the degree to
which the functions are nonlinear [8]. The incommensurability of accuracy and complexity then
necessitates another unmotivated heuristic.
Exhaustive Symbolic Regression (ESR; [9]) is designed to overcome these problems. The first
is solved by searching function space exhaustively, guaranteeing discovery of each and every
good function, and the second by replacing complexity heuristics with a precise measure of the
information content of the function called its description length.
To do an exhaustive search ESR generates every single possible function from some basis set
of operators up to a maximum complexity, where complexity is defined as the number of nodes
in the function’s tree representation. The operator basis set and maximum complexity are the
only things that must be specified by the user, although the maximum complexity is typically
set by the computational resources available. Generating all functions involves generating all
possible tree templates, where the nodes are labelled by their arity (number of arguments), and
decorating the trees by considering all permutations of the operators in the basis set with the
correct arity. We then simplify the functions and remove duplicates using a set of functioncomparison rules (tree reordering, parameter permutations, simplifications, reparametrisation invariance,
parameter combinations). This establishes the unique functions, which are all inequivalent to each
other and are representatives of sets of behaviourally identical but structurally different functions
(e.g. θx and x/θ, where θ is a free parameter). Finally, we find the maximum-likelihood values
of the free parameters appearing in the unique functions through nonlinear optimisation, and
broadcast these results to all other members of the equivalent sets using the Jacobians of the
transformations that relate them. Although the maximum likelihood values of all functions in
such a set must be identical, they may possess different description lengths (see below), and
hence our search for the lowest description length function ranges over these variants as well
as the unique functions. Full details may be found in [9].
This is a computationally expensive procedure due to the huge number of possible functions at
higher complexities. One’s computational budget therefore limits the complexity one can reach.
A typical limit is complexity 10, for which a full ESR run takes ∼200 CPU-hours (the scaling with
complexity is exponential, so even greatly enhanced computational resources could not extend
the maximum complexity by ≳ 1). This depends on the operator basis set (the more operators, the
more functions at given complexity and hence the lower the maximum achievable complexity),
and is necessarily approximate because the procedure is imperfectly parallelisable. Although the
5
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
parameter fitting is embarrassingly parallelisable (each function is treated completely separately
to all the others), the simplification steps must compare functions and hence cannot be done
in isolation. Note that only the parameter fitting is dataset-specific: the function generation and
simplification depend only on the operator set, allowing the user to benefit from publicly available
pre-computed function sets [10]. This reduces runtime by more than a factor of 2. For reference,
straight lines and power-laws have complexity 5, while the Navarro–Frenck–White (NFW; [11])
function describing halo density profiles (θ0(x(x + θ1)
θ2
)
−1
) has complexity 9: any function not
much more complex is within scope of ESR. The full ESR code is publicly available.1
(b) Benchmarking
To illustrate the advantage of a guaranteed search, this section demonstrates the unreliability
of stochastic algorithms. We take perhaps the simplest benchmark dataset (feynman_I_6_2a)
from the Penn Machine Learning Benchmarks dataset, as used in the SRBench competition [12]. This
comprises 105 datapoints generated from an unknown univariate function without scatter. In
addition to ESR we ran five state-of-the-art SR algorithms on the data: PySR, DataModeler [7],
FFX [13], QLattice [14] and Operon. The test was conducted under exam conditions, with each
algorithm given equal opportunity (full details in [9]).
Fig. 1 shows the Pareto front of MSE against complexity returned by each algorithm. The
most noticeable feature is the cliff in MSE produced by ESR at complexity 7, which indicates a
substantial improvement in the functional fit not seen by any other algorithm. The others can
discover the most accurate functions up to complexity 5, but beyond that find only marginally
improved solutions. On closer inspection it is seen that not only has ESR produced the bestfitting function, but in fact it has achieved the holy grail of SR which is to find the true function
that generated the data. The best complexity-7 function is y = θ1θ
x
2
0
, where θ0 = 0.6065 and
θ1 = 0.3989. This is remarkably similar to θ0 = 1/
√
e and θ1 = 1/
√
2π, suggesting that the data
were drawn from a standard normal distribution. Inputting these exact values for the parameters
yields an MSE = 3 × 10−33, which is 0 to within machine precision. That ESR did not achieve this
directly is due to its numerical tolerance in the parameter optimisation, which also explains why
slightly different MSE values are produced for the variants of the standard normal at complexities
8-10. The other SR algorithms gave no indication of this true structure, but simply produced
approximate fitting functions. The exception to this is Operon, which does in fact produce a
standard normal albeit overparametrised so that it appears at complexity 11.
The conclusion is that most SR algorithms fail even on very simple problems.
4. The minimum description length principle
With ESR we are guaranteed not to miss any function (built from the user-defined basis operator
set and up to the maximum complexity), but the problem of ranking them remains. Placing them
on the accuracy–complexity plane is sure to construct the true Pareto front (as in Fig. 1), but this
does not address the arbitrariness of complexity measure or incommensurability of complexity
and accuracy. We need a metric that can put accuracy and complexity on the same footing to
produce an objective, one-dimensional ranking.
Such an upgrade is analogous to that from maximum-likelihood to Bayesian inference.
Consider deciding between two models for some data. Simply considering the maximumlikelihood value, the more complex model is likely to be preferred, a procedure with no safeguard
against overfitting. But the model selection metric of Bayesian inference—the Bayesian evidence—
naturally penalises the more complex model; this model has lower prior values over the region of
parameter space that the data prefers, so is less predictive. This is reflected in approximations such
as the Bayesian information criterion (BIC), which trade off accuracy (maximum-likelihood value
Lˆ), with the number of free parameters p. The addition of another parameter must increase the
likelihood by some minimum amount for it to be warranted by the data. A heuristic measure such
1https://github.com/DeaglanBartlett/esr
6
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
Figure 1. Pareto front of mean square error against function complexity produced by six SR algorithms on the
feynman_I_6_2a benchmark dataset. Only ESR and Operon identify that the data is drawn from a standard normal,
and only ESR finds this function in its simplest form.
as the PySR score for combining accuracy and complexity [15] is akin to making up a function for
trading off Lˆ and p in ignorance of the evidence.
How can we generalise this argument to include also a penalisation for more complex
structures of functions, as well as more (and more finely-tuned) parameters as assessed by the
evidence? The answer is quite simple (perhaps the reader can already guess), but we will find it
instructive to come at it from the new angle of information theory.
Why do we want a functional fit in the first place? From an information-theoretic perspective it
is to provide a more efficient representation of the data: instead of communicating the full dataset
directly, one can hope to convey almost as much by communicating the function instead, which
requires many fewer bits. In the limiting case in which the function fits the data perfectly this
is completely lossless; otherwise one must either accept a loss of information or supplement the
communication with the residuals of the data around the function’s expectation, allowing the data
to be fully reconstructed. This is formalised in the minimum description length (MDL) principle,
which states that the goodness of each functional fit is quantified by the number of nats (log-e
bits) of information needed to convey the data with the help of the function, which is called the
description length [16,17]. The best function minimises this.
This requires communicating the function, or hypothesis H, and the residuals D|H where
D denotes the data, and may be written L(D) = L(H) + L(D|H), where L is the description
length operator. (L(D|H) may alternatively be considered the information loss, which acts as
a penalisation term.) How many nats are needed for each piece? Under an optimal coding
scheme called Shannon–Fano [18], the residual term is simply the minimum negative loglikelihood, describing the function’s accuracy on the data: L(D|H) = − log(Lˆ). L(H) comprises
the information content of the operators in the function and the values of any free parameters it
contains. If the function’s tree contains k nodes (what we are calling the function’s complexity)
populated by n distinct operators, there are n
k possible permutations corresponding to log(n
k
) =
k log(n) nats of information. Note that the value of this depends on the operator basis set: for
example it is smaller for tan(x) (if tan is included in the operator set) than sin(x)/cos(x) (if only
sin, cos and ÷ are included). Any natural numbers cj arising as part of the function simplification
process are encoded in P
j
log(cj ) further nats.
Finally we must encode the values of the maximum-likelihood parameters, ˆθi
. This is done by
choosing a precision ∆i with which to record the i
th parameter, thus representing it as |θi
|/∆i
7
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
increments encoded in log(|θi
|/∆i) nats. The larger ∆i
, the fewer nats are required to convey
|θi
|/∆i but the worse the hit to the likelihood when θi
is rounded to the nearest ∆i
. We choose the
∆i
that optimises this trade-off to produce the lowest total description length. This is found to be
at ∆i = (12/Iii)
1/2
, where I is the observed Fisher information matrix at the maximum-likelihood
point, Iij = −
∂
2
log L
∂θi∂θj



θˆ
. The sign of the parameter must be communicated with a further log(2)
nats. Putting this all together produces our final formula for the description length [9]:
L(D) = − log(L(
ˆθ)) + k log(n) −
p
2
log(3) +X
j
log(cj ) +Xp
i

1
2
log(Iii) + log(|
ˆθi
|)

, (4.1)
This is the quantity with which we score functions, lower being better.
We seem to have gone a long way from the Bayesian evidence discussed at the start of the
section. In fact we have unknowingly come full circle. Consider the generalisation of the evidence
Z (D|fi) for the probability of function fi given the data, including explicitly a prior P(fi) on the
function itself:
P (fi
|D) = 1
P(D)
Z
P (D|fi
, θi) P (θi
|fi) P (fi) dθi ≡
P (fi)
P(D)
Z (D|fi). (4.2)
Up to the overall normalisation term P(D) (the probability of getting the data from any
function), this simply multiplies the standard evidence by the function prior. Generalising the
implementation of Occam’s razor with regard to parametric complexity, it seems clear that in the
Bayesian context the penalisation of structural complexity must derive from this functional prior
term. Just as introducing additional parameters is apt to lower the prior over the high-likelihood
region of numerical parameter space, introducing additional operators should lower the prior
over the high-likelihood region of functional parameter space. The more operators one has, the
less likely any given combination. This is the “quite simple” solution alluded to above.
How does this relate to L(D)? To see the connection, calculate Z under the Laplace
approximation, which treats the posterior as a multidimensional Gaussian around the maximumposterior point:
− log P (fi
|D) ≃ − log P (fi) − log Z (D|fi) ≃ − log P (fi) − log H −ˆ
p
2
log 2π +
1
2
log det IˆH,
(4.3)
where the first relation neglects the overall normalising constant P(D) and the second
implements the Laplace approximation. H denotes the posterior and IˆH is the observed Fisher
information matrix for the posterior at the maximum posterior point. Eqs. 4.3 and 4.1 now look
suspiciously similar: both have terms depending on the maximum-likelihood value, the number
of free parameters and the observed Fisher information. It is simple to show [19] that the two
equations are effectively equivalent provided
− log P (fi) = k log n +
X
α
log cα. (4.4)
In other words, the description length is nothing more than (the negative log-likelihood of) the
Bayesian probability for the model under a specific choice of prior that penalises more structurally
complex functions.
This parallel also reveals a key freedom we have in calculating L(D). In Bayesian statistics
priors are subjective, representing our degree of belief before measurements are made. We may
therefore consider replacing the functional prior k log(n) + P
j
log(cj ) by something else. Why
would we want to do that? One reason could be that that prior is completely ignorant of the
structure of successful functions in the domain in question, for example those that describe “more
physical” solutions. As an example, suppose one is considering an oscillatory system and wishes
to assess whether some response goes as sin(x1) + sin(x2) or sin(sin(x1 + x2)). These possess
exactly the same operators, and hence have identical values of k log(n). However, our background
physics knowledge tells us that oscillatory systems are frequently described by sums of sines but
8
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
very rarely nested sines, so it is intuitively reasonable that our prior should favour the former
function.
This may be formalised by adapting an algorithm from language processing called the Katz
back-off model [20]. This takes in a training set of equations describing successful functions
within some pertinent domain, from which it learns probabilities of operator combinations. A
hyperparameter controls the length of these combinations, and another the minimum number of
occurrences of the combination in the training set for its probability to be directly calculated; for
combinations that do not appear that many times, the algorithm “backs off” to instead consider
a combination of one shorter length. This method therefore models prior probabilities for SRgenerated functions according to their similarity to those in the training set, the hope being 1)
that that set successfully describes something about the problem in question, and 2) that that
success is reflected in the prevalence of various operator combinations and hence may be learnt
by the model. We have constructed an implementation of the Katz model which may readily be
used as a drop-in replacement for the k log(n) + P
j
log(cj ) prior [19],2 where it is also shown to
perform well on a variety of benchmark tests.
5. Astrophysical applications
We want SR whenever we want a functional form for some relation. Sometimes, theory
(established or speculative) predicts a function and we want to know how good it is. At other
times, there is no prediction but we want either empirical inspiration for the governing law or a
fitting formula to summarise the phenomenology. This section describes three cases in the former
category, where concordance cosmology, gravity theory or inflationary model building makes
a prediction in astrophysics. We wish to see how well it performs relative to all other possible
predictions, and whether we could have recovered the theory directly from the empirical data.
(a) The cosmic expansion rate
Cosmology studies the overall structure and evolution of the universe. The standard cosmological
model is called ΛCDM because it posits that the present-day universe consists primarily of dark
energy (Λ) and cold dark matter (CDM). ΛCDM makes predictions called the Friedmann equations
for the expansion rate H of the universe as a function of time. They are written not in terms of time
directly, which is unobservable in cosmology, but rather in terms of redshift z, which describes
the amount by which photons’ wavelengths have been stretched due to expansion of the universe
en route to us. This acts as a proxy for time, with higher redshift meaning further into the past.
The simplest Friedmann equation, assuming matter is pressureless, is
H
2
(z) = H
2
0 (ΩΛ + Ωm(1 + z)
3
), (5.1)
where H0 ≡ H(0) is the current expansion rate and ΩΛ and Ωm are dimensionless quantities
describing the relative prevalence of dark energy and dark matter respectively. This equation can
be fitted to H(z) data and its parameters constrained. We could even calculate the goodness-of-fit
relative to alternative functions, for example with different equations of state for dark energy or
dark matter, through the Bayesian evidence. But this procedure necessarily specifies the models
under consideration a priori and can only rank them against each other; how can we see how
good the model is relative to all possible others?
SR offers the promise of determining the optimal form of H(z) directly from the data, without
requiring theorists to first come up with trial functions. In particular, ESR produces the complete
list of all possible (simple) H(z) functions and scores them on the data, producing an objective
ranking from which one can see how any given model fares relative to all others. This is the closest
one can come to absolute goodness-of-fit testing in a Bayesian context.
We use two datasets to do this [9]. The first, simpler, datasets involved cosmic chronometers
(CCs), stellar populations in galaxies that evolve passively such that their relative ages can
2https://github.com/DeaglanBartlett/katz
9
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
be determined from spectroscopy. Combined with the difference in their measured redshifts,
this affords a determination of ∆z/∆t, a discrete approximation to −(1 + z)H(z). This dataset
contains 32 points and derives from [21]. The second utilises Type Ia Supernovae (SNe), which are
explosions of massive stars and the end of their lives. SNe are “standardisable candles”, meaning
that, after some corrections relating to properties of their lightcurves, their luminosity is universal.
This enables calculation of their luminosity distances, from which may be calculated H(z). Here
we adopt 1590 SNe from the Pantheon+ sample, including the covariance matrix between SNe
which modulates the likelihood [22]. We simply plug these datasets separately into ESR (using
the k log(n) function prior) and turn the crank to produce all possible H(z) functions and their
DLs. We use the operator basis set {x ≡ 1 + z, θ, inv, +, −, ×, ÷, pow}.
The results are shown in Fig. 2: the left panel shows the 150 best functions found by ESR
colour-coded by their description length relative to the best function, compared to the CC data,
along with their residuals relative to Eq. 5.1 in the lower panel. The right panel shows the best DL
and log L discovered at each complexity in both datasets compared to the Friedmann equations.
“Λfluid” is a generalisation of Eq. 5.1 that allows matter to have pressure. For both datasets we
find that the MDL function has complexity 5, lower than the Friedmann equations at complexity
7. These MDL functions are
H
2
(z) = θ0(1 + z)
2
(CCs) , H2
(z) = θ0(1 + z)
1+z
(SNe), (5.2)
and are preferred over the simple Friedmann equation (Eq. 5.1) by 7.12 and 4.91 nats for the
CC and SN data respectively, which corresponds to probability ratios of 1240 and 136. We find
38 functions better than Eq. 5.1 for CCs and 36 for SNe. Note that while the Pareto front as
traditionally defined in terms of likelihood continues falling to high complexity, L(D) has a welldefined minimum beyond which the small gain in accuracy is swamped by the loss in simplicity.
This informs the user whether sufficiently complex functions have been considered: if not, the
minimum in L(D) will not yet have been seen.
What are we to make of the sub-optimality of the Friedmann equations? It is premature to
conclude that they are ruled out as the equations governing the data. It will be noticed that both
functions in Eq. 5.2 produce the same Taylor expansion for H(z) as the Friedmann equation up
to second order in z, making their behaviour (hence likelihood) very similar over the limited
redshift range of the data. They are however simpler functions, with smaller complexity terms in
their description lengths. It is indeed a feature of the MDL procedure that with limited quality
or quantity of data, the likelihood term is less important than the complexity terms and hence
simpler functions will be preferred. More data is therefore needed to settle definitively on the
best function(s), and the MDL formalism can tell us precisely what are the properties of the data
required [9].
(b) The radial acceleration relation
From the late-time expansion rate of the universe we turn to the internal dynamics of galaxies [23].
A puzzle here has been the enduring success of an alternative to the dark matter hypothesis on
which ΛCDM is based. This theory—Modified Newtonian Dynamics (MOND) [24–26]—posits
that the “missing gravity problem” in galaxies (e.g. that rotation curves become flat at large
galactocentric radius rather than declining in a Keplerian fashion) is due not to missing mass,
but rather to an alteration in either objects’ gravity or inertia at exceedingly low accelerations
below a new universal parameter a0 ≈ 1.2 × 10−10 m/s2
. This takes the form
⃗gobs = ν(gbar/a0) ⃗gbar, (5.3)
Where gbar is the acceleration sourced by the visible stars and gas in galaxies (“baryons”) and
gobs is the dynamical acceleration that drives objects’ motions. ν(x)—the “interpolating function”
(IF)—can take any form provided it satisfies the limits ν(x) → 1 for x ≫ 1 and ν(x) → x
−1/2
for x ≪ 1. These are called the “Newtonian” and “deep-MOND” limits; the former recovers
Newtonian dynamics at a ≫ a0 (e.g. in the Solar System), while the latter is the modification
10
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
50
100
150
200
250
300
H (z) / kms
−1Mpc
−1
10−1 100
z
−100
−50
0
50
(H (z) −
HΛCDM (z)) / kms
−1Mpc
−1
0
1
2
3
4
5
6
7
8
L (D) − MDL
0
20
40
∆
L(D)
Cosmic Chronometers
Pareto Front ΛCDM ΛFluid
4 5 6 7 8 9 10
Complexity
0
100
200
300
∆
L(D)
Pantheon+
0
5
10
15
|∆ log L|
0
10
20
30
|∆ log L|
Figure 2. Left: Top 150 ESR H(z) functions overplotted on the CC data (upper panel), and the residuals from Eq. 5.1
(lower panel). Right: Pareto fronts for both datasets with the literature standards (Friedmann equations) shown as separate
symbols. Reproduced from [9].
required to get flat rotation curves. Classic IF choices are the “Simple” (ν(x) = 1/2 + (1/4 +
1/x)
1/2
) and “RAR” (ν(x) = 1/(1 − exp(−
√
x))) functions.
This bizarre theory has astonishing successes and atrocious failures (see [27] for a recent
review). Its principal success is in predicting, and explaining the properties of, the “radial
acceleration relation” (RAR), which directly correlates gbar and gobs through measurements of
late-type galaxies’ HI and optical photometry and rotation curves through spectroscopy [28].
This relation has been found to possess a minute scatter and no correlated deviations from MOND
predictions [29], to satisfy four stringent criteria for “fundamentality” in galaxies’ radial dynamics
[30], and to evince a continuation to much lower accelerations through weak lensing [31]. We wish
to use SR to determine 1) whether the classic MOND IFs are optimal descriptions of the RAR data,
and 2) if not, whether the optimal functions possess the correct limits to be considered new IFs.
This will help determine the extent to which the RAR supports MOND.
We adopt the operator basis set {gbar, θ, exp, sqrt, square, inv, +, −, ×, ÷, pow}, and once again
simply plug the data into ESR using the k log(n) function prior to create and score the full set
of simple gobs(gbar) functions. The results are shown in Fig. 3. We find many functions better
than the MOND IFs. While most of these possess a Newtonian limit, very few possess a deepMOND limit: more commonly gobs tends to a constant at low gbar. Indeed, the fact that the
probability of a function is ∝ exp −L(D) (Sec. 4) lets us compute a probability over all functions
for certain limiting behaviour, which in this case is ∼1 for gobs → const as gbar → 0. However,
this is found also to be true on mock data generated using the RAR IF which explicitly possesses
such a limit [23]. Thus, while the best functions are not MONDian, one would not expect them
to be even if MOND did in fact generate the data, given the data’s limited dynamic range and
sizeable uncertainties. This is similar to the situation in Sec. 5(a), where the data could not have
been expected to return unambiguously the Friedmann equation even if it did in fact generate the
data. Better measurements are therefore required to know whether or not the RAR significantly
evidences MOND in its functional form.
(c) Inflation
For our final application we move to the very early universe [32]. An largely-accepted addon to ΛCDM postulates that within ∼ 10−32s of the Big Bang the universe underwent a
period of accelerated expansion known as inflation. This caused the universe to grow by a
factor ∼ e
60, smoothing it out and producing the seeds of structure growth through quantum
fluctuations. Inflation is driven by one or more scalar fields, called inflatons. The simplest scenario
is single-field, slow-roll inflation in which a single inflaton ϕ rolls down a shallow potential
11
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
Figure 3. Left: Pareto front identified by ESR for the RAR compared to the three most common literature fits. Right:
Logarithmic slopes of the 10 best ESR functions in the limit gbar → 0 (blue) and gbar → inf (red), and at the lower (cyan)
and upper (magenta) edges of the data. Where these depend on free parameters of the function they are shown as bands
indicating the 95% C.L. The MOND lower and upper slopes are shown by vertical dashed lines; especially the lower slope
is rarely satisfied by the best functions. Reproduced from [23].
gradient, generating a de Sitter-like epoch manifesting exponential expansion. This has a simple
Lagrangian and equation of motion
L = −
1
2
g
µν∂µϕ∂νϕ − V (ϕ) ⇒ ϕ¨ + 3Hϕ˙ + V
′
(ϕ) = 0. (5.4)
The unknown in these equations is the potential V (ϕ): the task for SR is to find the best such
functions.
Part of the problem with inflation—and the reason it is so amenable to SR—is that it is highly
underdetermined: there is a dazzling variety of theories of inflation [33], but effectively only
three numbers with which to test them. These are the amplitude and tilt of the primordial power
spectrum and upper limit on the tensor-to-scalar ratio, all measured from the cosmic microwave
background by the Planck satellite [34,35]:
As = (0.027 ± 0.0027)Mpl, ns = 0.9649 ± 0.0042, r < 0.028 (95% C.L.). (5.5)
This is the data with which we infer V (ϕ) using ESR. To explore the effect of operator basis set
we consider two possibilities: Set A: {ϕ, θ, inv, exp, log, sin, sqrt, square, cube, +, −, ×, ÷} and Set
B: {ϕ, θ, inv, exp, log, +, −, ×, ÷, pow}. Set A favours powers that are simply composed of sqrt,
square and cube, while set B increases flexibility with the general pow operator.
The Pareto front for basis set B is shown in Fig. 4 (left), using both the k log(n) (solid)
and Katz (dashed) function priors. For both basis sets the MDL function, at complexity 6, is
exp(− exp(exp(exp(ϕ)))), which, despite its byzantine structure, produces a very well-behaved
slow-roll region (Fig. 4, right). It is however unlikely to derive from a sensible underlying
particle physics theory of inflation. We therefore consider replacing the k log(n) prior, favouring
structurally simpler functions, with the Katz prior presented in Sec. 4, which favours functions
more similar to those in a training set. We choose as training set the Encyclopaedia Inflationaris,
containing almost all known literature theories of inflation and their corresponding inflaton
potentials [33]. The hope is that the “physicality” of these potentials (that is, their derivability
from a well-behaved particle physics theories) is reflected in the structure of their operator
combinations and may therefore be picked up by the Katz model. With this prior, the MDL
12
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
functions are θ0(θ1 + log(ϕ)
2
) for operator set A, and θ0ϕ
θ1/ϕ for set B. These are more reasonable
functions that may have a physical underpinning.
We can also use the full ranking afforded by ESR to compare the best functions to literature
standards. As an example, for operator set B with the k log(n) prior, the Starobinsky, quadratic
and quartic models place at ranks 1272, 8697 and 10839 respectively. Clearly, as in the previous
applications, SR is required to find the best functions.
Figure 4. Left: Pareto front for inflaton potentials using operator basis set B, using fiducial MDL (solid) or Katz back-off
model (dashed) function priors. Right: The potential favoured by the k log(n) prior, exp(− exp(exp(exp(ϕ)))), with the
slow-roll region over which inflation occurs shown by the red shaded band.
6. Upcoming developments and future work
SR is an exciting emergent field with much potential for the future. On a ∼10-year timescale it is
set to become a key method of the physicist’s (and scientist’s) toolbox, not only for finding fitting
functions in a principled and automated way but also for learning physical laws directly from
data. The symbolic side of ML will begin to see the successes and widespread application that the
“numerical” side already has.
We are planning a range of upgrades and further applications of ESR to realise this potential.
The most significant is a comprehensive set of upgrades to the ESR algorithm itself. This
involves a rewrite in Julia for automatic differentiation, expediting the parameter optimisation
and description length calculation, and uses a faster and more efficient scheme for simplifying
functions and removing duplicates. This will raise the complexity cap from 10 to ∼13. ESR 2.0
will also enable the modelling of multiple independent variables, and of snapping parameters to
integers, rational numbers and fundamental constants (e.g. c, e, π) where that would reduce the
description length. We are also planning ways to combine the virtues of ESR at low complexity
and GAs at higher complexity. This could include using the complete characterisation of function
space afforded by ESR to initialise GAs with the best-performing low-complexity functions, or to
optimise their hyperparameters through knowledge of the relations between such functions. The
MDL principle is independent of ESR and already finding use in GAs, where it has been shown
to improve performance [36].
On the practical side we have only scratched the surface of applications in astrophysics, let
alone science more generally. Our next priority is to use ESR to learn the optimal functional forms
of halo density profiles, both from N-body simulations and observations of galaxy kinematics.
This is a case where the standard, NFW profile was identified using the “by-eye” method
described in Sec. 1, and hence has scant justification despite being a crucial component of many
astrophysical and cosmological analyses. Similarly the so-called halo mass function and galaxy
mass and luminosity functions, describing the abundance of galaxies and dark matter halos as a
13
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
function of their luminosities and masses. The standard here is the Schechter function [37], which
is also highly likely to be improvable by a rigorous SR procedure. Indeed, any of the multifold
fitting functions prevalent in astrophysics may be upgraded by applying SR, and ESR provides a
simple and effective method for doing so.
7. Summary & Conclusion
Symbolic regression is the machine learning method that codifies and automates the practice of
empirical science, namely the creation of functions and equations describing data. While this
is traditionally done “by eye”, SR is a task for computers—affording an enormous increase in
processing power—if effective methods can be constructed.
SR is traditionally achieved by stochastic algorithms (e.g. genetic programming), with
candidate functions assessed by locating the Pareto front in accuracy and complexity and then
applying an additional ad hoc rule for selecting a single “best” function. I have argued that
such methods face two serious challenges: 1) they have a significant probability of failing to
find any given good function, and 2) the model selection procedure is unfounded in its arbitrary
definition of complexity and unjustified heuristic for breaking the Pareto front degeneracy. To
overcome the first I propose Exhaustive Symbolic Regression (ESR), and to overcome the second I
propose the Minimum Description Length (MDL) principle. ESR implements an efficient algorithm
for generating and optimising the parameters of all functions composed of a user-defined basis
set of operators up to a maximum complexity, guaranteeing discovery of all good solutions. MDL
measures functions’ quality with an information-theory-motivated metric that makes accuracy
and simplicity commensurable, affording a principled one-dimensional ranking. It is essentially
the Bayesian evidence plus a prior on functions that penalises those containing more, and more
varied, operators. An alternative prior, based on a Katz back-off language model, instead penalises
functions with combinations of operators that are rarer in a training set of equations.
I showcase ESR+MDL on three hot topics in astrophysics: the late-time expansion rate of the
universe, the effective behaviour of gravity in galaxies and the potential of the field driving
inflation. In each case, ESR discovers functions considerably superior to literature standards
(the Friedmann equation, MOND and Starobinsky, quadratic and quartic inflation respectively),
illustrating its ability to uncover effective symbolic representations of data purely empirically.
This bodes well for future discovery not only of optimal fitting functions, but, more ambitiously,
of physical laws directly from data.
SR is only just starting to take off. The near future will see an extensive overhaul of the ESR
algorithm, greatly improving its efficiency and capabilities. Synergy with genetic algorithms is
also promising, combining their advantages in different regimes of complexity. At the same time
there is a range of astrophysical (and other) fitting functions ripe for improvement. I suspect the
golden age of symbolic machine learning not to be far away.
Acknowledgements. I thank Deaglan Bartlett, Pedro Ferreira, Gabriel Kronberger, Lukas Kammerer and
Tomas Sousa for the collaborations on which this work is based. I am supported by a Royal Society University
Research Fellowship (grant no. 211046).
References
1. Sui C, Bartlett DJ, Pandey S, Desmond H, Ferreira PG, Wandelt BD. 2024 syren-new: Precise
formulae for the linear and nonlinear matter power spectra with massive neutrinos and
dynamical dark energy. arXiv e-prints p. arXiv:2410.14623. (10.48550/arXiv.2410.14623)
2. Turing AM. 1950 I.—COMPUTING MACHINERY AND INTELLIGENCE. Mind LIX, 433–460.
(10.1093/mind/LIX.236.433)
3. David E. 1989 Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley.
4. Haupt R, Haupt S. 2004 Practical genetic algorithms. Wyley 2nd edition.
5. Burlacu B, Kronberger G, Kommenda M. 2020 Operon C++: an efficient genetic programming
framework for symbolic regression. pp. 1562–1570. (10.1145/3377929.3398099)
14
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
6. Cranmer M, Sanchez-Gonzalez A, Battaglia P, Xu R, Cranmer K, Spergel D, Ho S. 2020
Discovering Symbolic Models from Deep Learning with Inductive Biases. NeurIPS 2020.
7. Evolved Analytics LLC. Data Modeler 9.5.1. Evolved analytics LLC. URL: www.
evolved-analytics.com; 2021.
8. Vladislavleva EJ, Smits GF, den Hertog D. 2009 Order of Nonlinearity as a Complexity
Measure for Models Generated by Symbolic Regression via Pareto Genetic Programming.
IEEE Transactions on Evolutionary Computation 13, 333–349. (10.1109/TEVC.2008.926486)
9. Bartlett DJ, Desmond H, Ferreira PG. 2024 Exhaustive Symbolic Regression. IEEE Transactions
on Evolutionary Computation 28, 950–964. (10.1109/TEVC.2023.3280250)
10. Bartlett DJ, Desmond H, Ferreira PG. 2022 Exhaustive Symbolic Regression Function Sets.
(10.5281/zenodo.7339113)
11. Navarro JF, Frenk CS, White SDM. 1996 The Structure of Cold Dark Matter Halos. ApJ 462,
563. (10.1086/177173)
12. La Cava W, Orzechowski P, Burlacu B, Olivetti de França F, Virgolin M, Jin Y, Kommenda M,
Moore JH. 2021 Contemporary Symbolic Regression Methods and their Relative Performance.
arXiv e-prints p. arXiv:2107.14351.
13. McConaghy T. 2011 pp. 235–260. In FFX: Fast, Scalable, Deterministic Symbolic Regression
Technology, pp. 235–260. New York, NY: Springer New York. (10.1007/978-1-4614-1770-5_13)
14. René Broløs K, Vieira Machado M, Cave C, Kasak J, Stentoft-Hansen V, Galindo Batanero V,
Jelen T, Wilstrup C. 2021 An Approach to Symbolic Regression Using Feyn. arXiv e-prints p.
arXiv:2104.05417. (10.48550/arXiv.2104.05417)
15. Cranmer M. 2020 PySR: Fast & Parallelized Symbolic Regression in Python/Julia.
(10.5281/zenodo.4041459)
16. Rissanen J. 1978 Modeling by shortest data description. Automatica 14, 465–471.
(https://doi.org/10.1016/0005-1098(78)90005-5)
17. Grünwald P, Roos T. 2019 Minimum Description Length Revisited. arXiv e-prints p.
arXiv:1908.08484.
18. Cover TM, Thomas JA. 1991 Elements of Information Theory. Wiley 2nd edition.
19. Bartlett DJ, Desmond H, Ferreira PG. 2023 Priors for symbolic regression. arXiv e-prints p.
arXiv:2304.06333. (10.48550/arXiv.2304.06333)
20. Katz SM. 1987 Estimation of probabilities from sparse data for the language model component
of a speech recognizer. IEEE Trans. Acoust. Speech Signal Process. 35, 400–401.
21. Moresco M et al.. 2022 Unveiling the Universe with emerging cosmological probes. Living
Reviews in Relativity 25, 6. (10.1007/s41114-022-00040-z)
22. Scolnic D et al.. 2021 The Pantheon+ Analysis: The Full Dataset and Light-Curve Release.
arXiv e-prints p. arXiv:2112.03863.
23. Desmond H, Bartlett DJ, Ferreira PG. 2023 On the functional form of the radial acceleration
relation. MNRAS 521, 1817–1831. (10.1093/mnras/stad597)
24. Milgrom M. 1983a A modification of the Newtonian dynamics as a possible alternative to the
hidden mass hypothesis. ApJ 270, 365–370. (10.1086/161130)
25. Milgrom M. 1983b A Modification of the Newtonian Dynamics - Implications for Galaxy
Systems. ApJ 270, 384. (10.1086/161132)
26. Milgrom M. 1983c A modification of the Newtonian dynamics - Implications for galaxies. ApJ
270, 371–389. (10.1086/161131)
27. Desmond H. 2025 Modified Newtonian Dynamics: Observational Successes and Failures.
arXiv e-prints p. arXiv:2505.21638. (10.48550/arXiv.2505.21638)
28. Lelli F, McGaugh SS, Schombert JM, Pawlowski MS. 2017 One Law to Rule Them All: The
Radial Acceleration Relation of Galaxies. ApJ 836, 152. (10.3847/1538-4357/836/2/152)
29. Desmond H. 2023 The underlying radial acceleration relation. MNRAS 526, 3342–3351.
(10.1093/mnras/stad2762)
30. Stiskalek R, Desmond H. 2023 On the fundamentality of the radial acceleration relation for
late-type galaxy dynamics. MNRAS 525, 6130–6145. (10.1093/mnras/stad2675)
31. Mistele T, McGaugh S, Lelli F, Schombert J, Li P. 2024 Radial acceleration relation of
galaxies with joint kinematic and weak-lensing data. JCAP 2024, 020. (10.1088/1475-
7516/2024/04/020)
32. Sousa T, Bartlett DJ, Desmond H, Ferreira PG. 2024 Optimal inflationary potentials. PRD 109,
083524. (10.1103/PhysRevD.109.083524)
33. Martin J, Ringeval C, Vennin V. 2013 Encyclopaedia Inflationaris. arXiv e-prints p.
arXiv:1303.3787. (10.48550/arXiv.1303.3787)
15
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
royalsocietypublishing.org/journal/rsta Phil. Trans. R. Soc. A 0000000
34. Planck Collaboration. 2020 Planck 2018 results. X. Constraints on inflation. AAP 641, A10.
(10.1051/0004-6361/201833887)
35. Galloni G, Bartolo N, Matarrese S, Migliaccio M, Ricciardone A, Vittorio N. 2023 Updated
constraints on amplitude and tilt of the tensor primordial spectrum. JCAP 2023, 062.
(10.1088/1475-7516/2023/04/062)
36. Burlacu B. 2023 GECCO’2022 Symbolic Regression Competition: Post-Analysis of the Operon
Framework. In Proceedings of the Companion Conference on Genetic and Evolutionary Computation
GECCO ’23 Companion p. 2412–2419 New York, NY, USA. Association for Computing
Machinery. (10.1145/3583133.3596390)
37. Schechter P. 1976 An analytic expression for the luminosity function for galaxies.. ApJ 203,
297–306. (10.1086/154079)
"""

# Initialiser le résumeur
summarizer = GenerativeSummarizer()

# Générer le résumé
result = summarizer.summarize(long_text)

# Afficher le résultat
print("Résumé :", result['summary'])
print("Méthode :", result['method'])
print("Confiance :", result['confidence'])
