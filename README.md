# Careless

## Procedures for Computing Indices of Careless Responding

When taking online surveys, participants sometimes respond to items without regard to their content. These types of responses, referred to as careless or insufficient effort responding, constitute significant problems for data quality, leading to distortions in data analysis and hypothesis testing, such as spurious correlations. The 'R' package 'careless' provides solutions designed to detect such careless / insufficient effort responses by allowing easy calculation of indices proposed in the literature. It currently supports the calculation of longstring, even-odd consistency, psychometric synonyms/antonyms, Mahalanobis distance, and intra-individual response variability (also termed inter-item standard deviation). For a review of these methods, see [Curran (2016)](https://www.sciencedirect.com/science/article/abs/pii/S0022103115000931?via%3Dihub)

## Description
Careless or insufficient effort responding in surveys, i.e. responding to items without regard to their
content, is a common occurence in surveys. These types of responses constitute significant prob-
lems for data quality leading to distortions in data analysis and hypothesis testing, such as spurious
correlations. The R package careless provides solutions designed to detect such careless / insuffi-
cient effort responses by allowing easy calculation of indices proposed in the literature. It currently
supports the calculation of Longstring, Even-Odd Consistency, Psychometric Synonyms/Antonyms,
Mahalanobis Distance, and Intra-individual Response Variability (also termed Inter-item Standard
Deviation).

## Statistical Outlier Function
* **mahad** computes Mahalanobis Distance, which gives the distance of a data point relative to the
center of a multivariate distribution.
## Consistency Indices
* **evenodd** computes the Even-Odd Consistency Index. It divides unidimensional scales using
an even-odd split; two scores, one for the even and one for the odd subscale, are then computed
as the average response across subscale items. Finally, a within-person correlation is computed
based on the two sets of subscale scores for each scale.
* **psychsyn** computes the Psychometric Synonyms Index, or, alternatively, the Psychometric
Antonyms Index. Psychometrical synonyms are item pairs which are correlated highly posi-
tively, whereas psychometric antonyms are item pairs which are correlated highly negatively.
A within-person correlation is then computed based on these item pairs.
* **psychant** is a convenience wrapper for psychsyn that computes psychological antonyms.
* **psychsyn_critval** is a helper designed to set an adequate critical value (i.e. magnitude of
correlation) for the psychometric synonyms/antonyms index.
## Response Pattern Functions
* **longstring** computes the longest (and optionally, average) length of consecutive identical
responses given.
* **irv** computes the Intra-individual Response Variability (IRV), the "standard deviation of responses across a set of consecutive item responses for an individual" [Dunn et al. 2018](https://link.springer.com/article/10.1007/s10869-016-9479-0)
