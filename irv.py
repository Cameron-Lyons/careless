"""
The IRV is the "standard deviation of responses across a set of consecutive item responses for
an individual" (Dunn, Heggestad, Shanock, & Theilgard, 2018, p. 108). By default, the IRV is
calculated across all columns of the input data. Additionally it can be applied to different subsets
of the data. This can detect degraded response quality which occurs only in a certain section of the
questionnaire (usually the end). Whereas Dunn et al. (2018) propose to mark persons with low IRV
scores as outliers - reflecting straightlining responses, Marjanovic et al. (2015) propose to mark
persons with high IRV scores - reflecting highly random responses
"""
