# Automated Data Analysis Report

## Introduction
This report provides a summary of the dataset, including key statistics, visualizations, and insights.

## Summary Statistics
|       |   book_id |   goodreads_book_id |     best_book_id |         work_id |   books_count |         isbn13 |   original_publication_year |   average_rating |    ratings_count |   work_ratings_count |   work_text_reviews_count |   ratings_1 |   ratings_2 |   ratings_3 |      ratings_4 |       ratings_5 |
|:------|----------:|--------------------:|-----------------:|----------------:|--------------:|---------------:|----------------------------:|-----------------:|-----------------:|---------------------:|--------------------------:|------------:|------------:|------------:|---------------:|----------------:|
| count |  10000    |     10000           |  10000           | 10000           |    10000      | 9415           |                    9979     |     10000        |  10000           |      10000           |                  10000    |    10000    |    10000    |     10000   | 10000          | 10000           |
| mean  |   5000.5  |         5.2647e+06  |      5.47121e+06 |     8.64618e+06 |       75.7127 |    9.75504e+12 |                    1981.99  |         4.00219  |  54001.2         |      59687.3         |                   2919.96 |     1345.04 |     3110.89 |     11475.9 | 19965.7        | 23789.8         |
| std   |   2886.9  |         7.57546e+06 |      7.82733e+06 |     1.17511e+07 |      170.471  |    4.42862e+11 |                     152.577 |         0.254427 | 157370           |     167804           |                   6124.38 |     6635.63 |     9717.12 |     28546.4 | 51447.4        | 79768.9         |
| min   |      1    |         1           |      1           |    87           |        1      |    1.9517e+08  |                   -1750     |         2.47     |   2716           |       5510           |                      3    |       11    |       30    |       323   |   750          |   754           |
| 25%   |   2500.75 |     46275.8         |  47911.8         |     1.00884e+06 |       23      |    9.78032e+12 |                    1990     |         3.85     |  13568.8         |      15438.8         |                    694    |      196    |      656    |      3112   |  5405.75       |  5334           |
| 50%   |   5000.5  |    394966           | 425124           |     2.71952e+06 |       40      |    9.78045e+12 |                    2004     |         4.02     |  21155.5         |      23832.5         |                   1402    |      391    |     1163    |      4894   |  8269.5        |  8836           |
| 75%   |   7500.25 |         9.38223e+06 |      9.63611e+06 |     1.45177e+07 |       67      |    9.78083e+12 |                    2011     |         4.18     |  41053.5         |      45915           |                   2744.25 |      885    |     2353.25 |      9287   | 16023.5        | 17304.5         |
| max   |  10000    |         3.32886e+07 |      3.55342e+07 |     5.63996e+07 |     3455      |    9.79001e+12 |                    2017     |         4.82     |      4.78065e+06 |          4.94236e+06 |                 155254    |   456191    |   436802    |    793319   |     1.4813e+06 |     3.01154e+06 |

## Missing Values
|                           |    0 |
|:--------------------------|-----:|
| book_id                   |    0 |
| goodreads_book_id         |    0 |
| best_book_id              |    0 |
| work_id                   |    0 |
| books_count               |    0 |
| isbn                      |  700 |
| isbn13                    |  585 |
| authors                   |    0 |
| original_publication_year |   21 |
| original_title            |  585 |
| title                     |    0 |
| language_code             | 1084 |
| average_rating            |    0 |
| ratings_count             |    0 |
| work_ratings_count        |    0 |
| work_text_reviews_count   |    0 |
| ratings_1                 |    0 |
| ratings_2                 |    0 |
| ratings_3                 |    0 |
| ratings_4                 |    0 |
| ratings_5                 |    0 |
| image_url                 |    0 |
| small_image_url           |    0 |

## Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

## Outliers Detection
|                           |    0 |
|:--------------------------|-----:|
| book_id                   |    0 |
| goodreads_book_id         |  345 |
| best_book_id              |  357 |
| work_id                   |  601 |
| books_count               |  844 |
| isbn13                    |  556 |
| original_publication_year | 1031 |
| average_rating            |  158 |
| ratings_count             | 1163 |
| work_ratings_count        | 1143 |
| work_text_reviews_count   | 1005 |
| ratings_1                 | 1140 |
| ratings_2                 | 1156 |
| ratings_3                 | 1149 |
| ratings_4                 | 1131 |
| ratings_5                 | 1158 |

![Outliers](outliers.png)

## Distribution of Data
![Distribution](distribution_.png)

## Conclusion
This analysis highlights key patterns and insights in the dataset.
## Data Story
### Narrative Based on Data Analysis of Books Dataset

#### Overview
The dataset under analysis comprises 10,000 entries, each representing a unique book. Several key attributes are documented, including identifiers, ratings, publication details, and the number of books by each author. This analysis aims to summarize the dataset's characteristics, identify missing values, examine correlations, and highlight potential outliers.

#### Summary Statistics
The dataset features a range of metrics that indicate the popularity and reception of the books. The average book has received about 75.7 total ratings, with a mean average rating of approximately 3.45 (derived from the ratings distribution). The ratings are heavily skewed toward the higher end, with a mean of 1345 ratings at 1 star and a maximum of over 456,191 for the highest-rated book. This suggests a significant variance in how books are received across different readers.

The ratings distribution shows that most books receive a good share of 4 and 5-star ratings, which is positive. However, the standard deviation for ratings is notably high, indicating that while many books are well-received, there are significant numbers that do not resonate with readers.

#### Missing Values
The analysis reveals several columns with missing entries. Notably, the `isbn` and `isbn13` fields have a considerable number of missing values (700 and 585, respectively). Additionally, there are missing values for the `original_publication_year` and `original_title` fields, with 21 and 585 missing entries, respectively. This indicates that while most entries are complete, there are critical attributes pertaining to the books' identification and publication history that may need attention, especially for comprehensive data analysis and retrieval.

#### Correlation Insights
The correlation matrix indicates several significant relationships within the dataset. Notably, `ratings_count` and `work_ratings_count` exhibit strong positive correlations with each other and the individual star ratings (ratings_1 through ratings_5). This suggests that books that receive a higher number of ratings also tend to have higher average ratings, reinforcing the idea that popularity correlates with perceived quality.

Interestingly, `books_count` shows a negative correlation with `average_rating`, hinting that books with more editions or versions might not necessarily be rated higher, potentially due to reader bias or market saturation.

#### Outlier Detection
Outlier analysis reveals several fields with notable extremes. For instance, `goodreads_book_id` and `best_book_id` each have hundreds of outlier entries, suggesting that there are a few books that stand out dramatically in terms of their ratings or number of reviews. The `average_rating` field also shows 158 entries that fall outside the expected range, indicating that there are some books that are either exceptionally well-received or poorly rated.

The presence of outliers in fields like `ratings_count` and `work_text_reviews_count suggests that a small number of books receive a disproportionately high level of attention and feedback from readers, which could skew statistical analyses if not addressed.

#### Conclusion
This analysis of the books dataset highlights the diverse responses that books receive from readers, showcasing a mix of popular titles and lesser-known works. The dataset is relatively complete, though attention to missing values and outliers is warranted to ensure a robust analysis. The strong correlations observed among ratings and the presence of significant outliers emphasize the dynamic nature of reader engagement in the literary market. Going forward, addressing missing values and potential biases introduced by outliers will be crucial for deeper insights and more accurate predictive modeling in the realm of book ratings and recommendations.