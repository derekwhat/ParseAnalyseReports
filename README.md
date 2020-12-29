# ParseAnalyseReports

Incredibly rough code.
Project was done during my exchange in Tokyo, at Tokyo Insitute of Technology. 

## Navigated Japanese Regulatory site containing English Corporate Governance Reports (CSG)
- Used Selenium and code to convert pdf into html to parse for date published 
- Each company had their own reporting format, and could change over the years
- data had to be standardized
- Resuling file contained: issuer info, and time of all releases was created
  - this was made into .csv file for tracking
  
## STatistical analysis of reporting times vs other corporate data
- Used Scipy and pandas to non-parametrically confirm relationships between variables
- Used Statsmodel and pandas to create different regression models to test statistcal relationships
  - accounted for common differentiating factors between company (size, industry, etc.) and used indicator variables
  - regresed on the impact English CSG reports had on foreign investors (specifically the % of the firm held by foreign investors)
  - accounted for the possibility of endongenity, multi-linearcollinarity, and scaling issues
- then wrote analysis and business world reasoning to support the analysis
