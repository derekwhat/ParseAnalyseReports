# ParseAnalyseReports

Incredibly rough code.
Project was done during my exchange in Tokyo, at Tokyo Insitute of Technology.
### Accomplished:
- Navigating websites
  - selenium
- Scraping data from websites and parsing data from pdf files into html
  - requests, pdfminer, beautiful soup, regular expressions
- Preparing data for exporting into analysis and combining files of data
  - pandas, sql
- Conducting statistical analysis on data
  - statsmodel

## Navigated Japanese Regulatory site containing English Corporate Governance Reports (CSG) (*Upload Date Scaper.py*)
- Used Selenium and code to convert pdf into html to parse for date published 
- Each company had their own reporting format, and could change over the years
- data had to be standardized
- Resuling file contained: issuer info, and time of all releases was created
  - this was made into .csv file for tracking
  
## STatistical analysis of reporting times vs other corporate data (*Stata data.py*)
- Used Scipy and pandas to non-parametrically confirm relationships between variables
- Used Statsmodel and pandas to create different regression models to test statistcal relationships
  - accounted for:
    - common differentiating factors between company (size, industry, etc.)
    - tracking companies through the use of an indiciator variable matrix
    - endogenity by regressing on the output variable to predict the input variable
    - multi-linearcollinarty by removing redundant variables
    - scaling issues by taking the log of differentiating factors, or dividing by a certain multiple of 10 for each relevant factor
    - 
  - regresed on the impact English CSG reports had on foreign investors (specifically the % of the firm held by foreign investors)
  - accounted for the possibility of endongenity, multi-linearcollinarity, and scaling issues
- then wrote analysis and reasoning based in financial theory to support bridge the gap between model and real world


# Research Poster presented
![Results of Analysis](https://github.com/derekwhat/ParseAnalyseReports/blob/main/YSEP%20poster.pdf)
