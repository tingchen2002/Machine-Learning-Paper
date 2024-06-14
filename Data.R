library(nhanesA)
library("glmnetcr")
library("dplyr")
#Get the required data, clean it up and adjust observations if needed (make categories ordinal factors)
#For cycle 2015-2016 Questionnaire data
DIQ_I  <- nhanes('DIQ_I')
DIQ_I <- DIQ_I %>%
  select(SEQN, DIQ050, DIQ172)
DIQ_I <- DIQ_I[DIQ_I$DIQ050 != "Don't know", ]
DIQ_I <- DIQ_I[DIQ_I$DIQ172 != "Don't know", ]
DIQ_I$DIQ050 <- as.numeric(DIQ_I$DIQ050 == "Yes")
DIQ_I$DIQ172 <- as.numeric(DIQ_I$DIQ172 == "Yes")

MCQ_I  <- nhanes('MCQ_I')
MCQ_I <- MCQ_I %>%
  select(SEQN,MCQ010, MCQ053, MCQ080, MCQ365B, MCQ365A, MCQ365C, MCQ370A, MCQ370C,MCQ370D, MCQ370B)
ctc <- c("MCQ010", "MCQ053", "MCQ080", "MCQ365B", "MCQ365A", "MCQ365C", "MCQ370A", "MCQ370C", "MCQ370D", "MCQ370B")
for(col_name in ctc) {
  MCQ_I[[col_name]] <- as.numeric(MCQ_I[[col_name]] == "Yes")
}
MCQ_I <- na.omit(MCQ_I)

BPQ_I  <- nhanes('BPQ_I')
BPQ_I <- BPQ_I %>%
  select(SEQN, BPQ020)
BPQ_I$BPQ020 <- as.numeric(BPQ_I$BPQ020 == "Yes")

HEQ_I  <- nhanes('HEQ_I')
HEQ_I <- HEQ_I %>%
  select(SEQN,HEQ010, HEQ030)
HEQ_I$HEQ010 <- as.numeric(HEQ_I$HEQ010 == "Yes")
HEQ_I$HEQ030 <- as.numeric(HEQ_I$HEQ030 == "Yes")


PAQ_I  <- nhanes('PAQ_I')
PAQ_I <- PAQ_I %>%
  select(SEQN,PAQ650, PAQ665)
PAQ_I$PAQ650 <- as.numeric(PAQ_I$PAQ650 == "Yes")
PAQ_I$PAQ665 <- as.numeric(PAQ_I$PAQ665 == "Yes")

SMQ_I  <- nhanes('SMQ_I')
SMQ_I <- SMQ_I %>%
  select(SEQN,SMQ020)
SMQ_I$SMQ020 <- as.numeric(SMQ_I$SMQ020 == "Yes")


DIQ_I  <- nhanes('DIQ_I')
DIQ_I <- DIQ_I %>%
  select(SEQN,DIQ010,DIQ050)
DIQ_I$DIQ010 <- as.numeric(DIQ_I$DIQ010 == "Yes")
DIQ_I$DIQ050 <- as.numeric(DIQ_I$DIQ050 == "Yes")

OHQ_I  <- nhanes('OHQ_I')
OHQ_I <- OHQ_I %>%
  select(SEQN,OHQ030,OHQ770,OHQ845)
OHQ_I$OHQ770 <- as.numeric(OHQ_I$OHQ770 == "Yes")
unique_obs <- unique(OHQ_I$OHQ030)
responses <- c(
  "6 months or less",
  "More than 3 years, but not more than 5 years ago",
  "More than 2 years, but not more than 3 years ago",
  "More than 6 months, but not more than 1 year ago",
  "More than 5 years ago",
  "More than 1 year, but not more than 2 years ago",
  "Don't know",
  "Refused"
)
numeric_values <- c(
  "6 months or less" = 1,
  "More than 6 months, but not more than 1 year ago" = 2,
  "More than 1 year, but not more than 2 years ago" = 3,
  "More than 2 years, but not more than 3 years ago" = 4,
  "More than 3 years, but not more than 5 years ago" = 5,
  "More than 5 years ago" = 6,
  "Don't know" = NA,
  "Refused" = NA
)
OHQ_I$OHQ030 <- numeric_values[OHQ_I$OHQ030]
unique(OHQ_I$OHQ845)
responses <- c("Excellent", "Very good", "Good", "Fair", "Poor", "Don't know")
numeric_values <- c("Excellent" = 5, "Very good" = 4, "Good" = 3, "Fair" = 2, "Poor" = 1, "Don't know" = NA)
OHQ_I$OHQ845 <- numeric_values[OHQ_I$OHQ845]

DPQ_I	  <- nhanes('DPQ_I')
DPQ_I <- DPQ_I[, !colnames(DPQ_I) %in% "DPQ100"]
unique(DPQ_I$DPQ010)
columns_to_categorize <- setdiff(names(DPQ_I), "SEQN")
responses <- c("Not at all", "Several days", "More than half the days", "Nearly every day", "Don't know", "Refused")
numeric_values <- c("Not at all" = 0, "Several days" = 1, "More than half the days" = 2, "Nearly every day" = 3, "Don't know" = NA, "Refused" = NA)
for (col in columns_to_categorize) {
  DPQ_I[[col]] <- numeric_values[DPQ_I[[col]]]
}

columns_sum <- c("DPQ010","DPQ020","DPQ030","DPQ040","DPQ050","DPQ060","DPQ070","DPQ080","DPQ090")
DPQ_I$sum <- rowSums(DPQ_I[, columns_sum], na.rm = TRUE)

DPQ_I$depression <- cut(DPQ_I$sum, 
                                 breaks = c(-Inf, 4, 9, 14, 27),
                                 labels = c("normal", "mild", "moderate", "moderately severe to severe"))

DPQ_I$depression_numeric <- as.integer(DPQ_I$depression)

INQ_I  <- nhanes('INQ_I')
INQ_I <- INQ_I %>%
  select(SEQN,IND235,IND310, INQ060, INQ140, INQ150, INQ300)
unique(INQ_I$IND235)
responses <- c(
  "$5,400 - $6,249", "$0 - $399", "$2,100 - $2,899", "$2,900 - $3,749", "$8,400 and over",
  NA, "$4,600 - $5,399", "$1,650 - $2,099", "$800 - $1,249", "$1,250 - $1,649", "Don't know",
  "$3,750 - $4,599", "$6,250 - $8,399", "$400 - $799", "Refused"
)
numeric_values <- c(
  "$0 - $399" = 1,
  "$400 - $799" = 2,
  "$800 - $1,249" = 3,
  "$1,250 - $1,649" = 4,
  "$1,650 - $2,099" = 5,
  "$2,100 - $2,899" = 6,
  "$2,900 - $3,749" = 7,
  "$3,750 - $4,599" = 8,
  "$4,600 - $5,399" = 9,
  "$5,400 - $6,249" = 10,
  "$6,250 - $8,399" = 11,
  "$8,400 and over" = 12,
  "Don't know" = NA,
  "Refused" = NA
)
INQ_I$IND235 <- numeric_values[as.character(INQ_I$IND235)]
unique(INQ_I$IND310)
responses <- c(
  NA, "$3,001 - $5,000", "$0 - $3,000", "$15,001 - $20,000", "Refused", "$10,001 - $15,000",
  "$5,001 - $10,000", "Don't know"
)

numeric_values <- c(
  "$0 - $3,000" = 1500,
  "$3,001 - $5,000" = 4000,
  "$5,001 - $10,000" = 7500,
  "$10,001 - $15,000" = 12500,
  "$15,001 - $20,000" = 17500,
  "Don't know" = NA,
  "Refused" = NA
)

INQ_I$IND310 <- numeric_values[as.character(INQ_I$IND310)]
INQ_I <- INQ_I %>%
  mutate(
    INQ060 = as.numeric(INQ060 == "Yes"),
    INQ140 = as.numeric(INQ140 == "Yes"),
    INQ150 = as.numeric(INQ150 == "Yes"),
    INQ300 = as.numeric(INQ300 == "Yes")
  )


OCQ_I  <- nhanes('OCQ_I')
OCQ_I <- OCQ_I %>%
  select(SEQN,OCQ180,OCD270)

SLQ_I  <- nhanes('SLQ_I')
SLQ_I <- SLQ_I %>%
  select(SEQN,SLD012, SLQ050)
SLQ_I$OHQ770 <- as.numeric(SLQ_I$SLQ050 == "Yes")

ALQ_I  <- nhanes('ALQ_I')
ALQ_I <- ALQ_I %>%
  select(SEQN,ALQ101, ALQ130, ALQ151)
ALQ_I$ALQ101 <- as.numeric(ALQ_I$ALQ101 == "Yes")
ALQ_I$ALQ151 <- as.numeric(ALQ_I$ALQ151 == "Yes")

DUQ_I  <- nhanes('DUQ_I')
DUQ_I <- DUQ_I %>%
  select(SEQN,DUQ200, DUQ370)
DUQ_I$DUQ200 <- as.numeric(DUQ_I$DUQ200 == "Yes")
DUQ_I$DUQ370 <- as.numeric(DUQ_I$DUQ370 == "Yes")

CBQ_I  <- nhanes('CBQ_I')

FSQ_I  <- nhanes('FSQ_I')
FSQ_I <- FSQ_I %>%
  select(SEQN,FSD032A, FSDHH, FSDAD, FSDCH, FSD151)
unique(FSQ_I$FSD032A)

responses <- c("Never true", "Sometimes true", "Often true", "Don't know")
numeric_values <- c(
  "Never true" = 1,
  "Sometimes true" = 2,
  "Often true" = 3,
  "Don't know" = NA
)

FSQ_I$FSD032A <- numeric_values[as.character(FSQ_I$FSD032A)]
unique(FSQ_I$FSDHH)
responses <- c(
  "HH full food security: 0",
  "HH marginal food security: 1-2",
  "HH low food security: 3-5 (HH w/o child) / 3-7 (HH w/ child)",
  "HH very low food security: 6-10 (HH w/o child) / 8-18 (HH w/ child)"
)

ordinal_values <- c(
  "HH full food security: 0" = 1,
  "HH marginal food security: 1-2" = 2,
  "HH low food security: 3-5 (HH w/o child) / 3-7 (HH w/ child)" = 3,
  "HH very low food security: 6-10 (HH w/o child) / 8-18 (HH w/ child)" = 4
)

FSQ_I$FSDHH <- ordinal_values[as.character(FSQ_I$FSDHH)]
unique(FSQ_I$FSDAD)
responses <- c(
  "AD full food security: 0",
  "AD marginal food security: 1-2",
  "AD low food security: 3-5",
  "AD very low food security: 6-10",
  NA
)
ordinal_values <- c(
  "AD full food security: 0" = 1,
  "AD marginal food security: 1-2" = 2,
  "AD low food security: 3-5" = 3,
  "AD very low food security: 6-10" = 4,
  NA 
)
FSQ_I$FSDAD <- ordinal_values[as.character(FSQ_I$FSDAD)]
unique(FSQ_I$FSDCH)
responses <- c(
  "CH full or marginal food security: 0",
  "CH marginal food security: 1",
  "CH low food security: 2-4",
  "CH very low food security: 5-8",
  NA
)

ordinal_values <- c(
  "CH full or marginal food security: 0" = 1,
  "CH marginal food security: 1" = 2,
  "CH low food security: 2-4" = 3,
  "CH very low food security: 5-8" = 4,
  NA
)
FSQ_I$FSDCH <- ordinal_values[as.character(FSQ_I$FSDCH)]

FSQ_I$FSD151 <- as.numeric(FSQ_I$FSD151 == "Yes")

CDQ_I  <- nhanes('CDQ_I')
CDQ_I <- CDQ_I %>%
  select(SEQN,CDQ001, CDQ010)
CDQ_I$CDQ001 <- as.numeric(CDQ_I$CDQ001 == "Yes")
CDQ_I$CDQ010 <- as.numeric(CDQ_I$CDQ010 == "Yes")

#2015-2016 cycle Examination data

BPX_I  <- nhanes('BPX_I')
BPX_I <- BPX_I %>%
  select(SEQN,BPXDI1, BPXML1, BPXSY1)

BMX_I	  <- nhanes('BMX_I')
BMX_I	 <- BMX_I %>%
  select(SEQN,BMXBMI, BMXHT,BMXWT)

OHXREF_I		<- nhanes('OHXREF_I')
OHXREF_I <- OHXREF_I	 %>%
  select(SEQN,OHARNF, OHAREC)
unique(OHXREF_I$OHAREC)
responses <- c(
  "Continue your regular routine care",
  "See a dentist at your earliest convenience",
  "See a dentist within the next 2 weeks",
  "See a dentist immediately",
  NA
)

# Define the ordinal values corresponding to each response
ordinal_values <- c(
  "Continue your regular routine care" = 1,
  "See a dentist at your earliest convenience" = 2,
  "See a dentist within the next 2 weeks" = 3,
  "See a dentist immediately" = 4,
  NA 
)

OHXREF_I$OHAREC <- ordinal_values[as.character(OHXREF_I$OHAREC)]
OHXREF_I$OHARNF <- as.numeric(OHXREF_I$OHARNF == "Yes")

#Cycle 2015-2016 demographics data
DEMO_I	  <- nhanes('DEMO_I')
DEMO_I	 <- DEMO_I	 %>%
  select(SEQN,DMDEDUC2,DMDFMSIZ	,DMDHHSIZ, DMDHHSZA	,DMDHHSZB, DMDHHSZE, INDFMIN2, INDHHIN2,RIAGENDR, RIDAGEYR, SDMVPSU, SDMVSTRA, WTMEC2YR, DMDCITZN,INDFMPIR	)
unique(DEMO_I$DMDMARTL)

unique(DEMO_I$DMDCITZN)
responses <- c(
  "Citizen by birth or naturalization",
  "Not a citizen of the US",
  "Don't Know",
  "Refused",
  NA
)

numeric_values <- c(
  "Citizen by birth or naturalization" = 0,
  "Not a citizen of the US" = 1,
  "Don't Know" = NA,
  "Refused" = NA,
  NA
)
DEMO_I$DMDCITZN <- numeric_values[as.character(DEMO_I$DMDCITZN)]

DEMO_I$RIAGENDR <- as.numeric(DEMO_I$RIAGENDR == "Male")
unique(DEMO_I$DMDEDUC2)
responses <- c(
  "Less than 9th grade",
  "9-11th grade (Includes 12th grade with no diploma)",
  "High school graduate/GED or equivalent",
  "Some college or AA degree",
  "College graduate or above",
  "Don't Know",
  NA
)

# Define the ordinal values corresponding to each response
ordinal_values <- c(
  "Less than 9th grade" = 1,
  "9-11th grade (Includes 12th grade with no diploma)" = 2,
  "High school graduate/GED or equivalent" = 3,
  "Some college or AA degree" = 4,
  "College graduate or above" = 5,
  "Don't Know" = NA,
  NA 
)

DEMO_I$DMDEDUC2 <- ordinal_values[as.character(DEMO_I$DMDEDUC2)]
unique(DEMO_I$DMDHHSIZ)
DEMO_I$DMDHHSIZ <- gsub("7 or more people in the Household", "7", DEMO_I$DMDHHSIZ)
DEMO_I$DMDFMSIZ <- gsub("7 or more people in the Family", "7", DEMO_I$DMDFMSIZ)
DEMO_I$DMDHHSZA <- gsub("3 or more", "3", DEMO_I$DMDHHSZA)
DEMO_I$DMDHHSZB <- gsub("4 or more", "4", DEMO_I$DMDHHSZB)

unique(DEMO_I$INDFMIN2)
responses <- c(
  "$ 0 to $ 4,999", "$ 5,000 to $ 9,999", "$10,000 to $14,999",
  "$15,000 to $19,999", "$20,000 to $24,999", "$25,000 to $34,999",
  "$35,000 to $44,999", "$45,000 to $54,999", "$55,000 to $64,999",
  "$65,000 to $74,999", "$75,000 to $99,999", "$100,000 and Over",
  "Under $20,000", "Refused", "Don't know", NA
)

numeric_values <- c(
  "$ 0 to $ 4,999" = 1, "$ 5,000 to $ 9,999" = 2, "$10,000 to $14,999" = 3,
  "$15,000 to $19,999" = 4, "$20,000 to $24,999" = 5, "$25,000 to $34,999" = 6,
  "$35,000 to $44,999" = 7, "$45,000 to $54,999" = 8, "$55,000 to $64,999" = 9,
  "$65,000 to $74,999" = 10, "$75,000 to $99,999" = 11, "$100,000 and Over" = 12,
  "Under $20,000" = 13, "Refused" = NA, "Don't know" = NA, NA
)
DEMO_I$INDFMIN2 <- numeric_values[as.character(DEMO_I$INDFMIN2)]
DEMO_I$INDHHIN2 <- numeric_values[as.character(DEMO_I$INDHHIN2)]


#Merge all the dataframes
df <- PAQ_I
dataframes <- list(DIQ_I, MCQ_I, BPQ_I, SMQ_I, DIQ_I, OHQ_I, DPQ_I, OCQ_I, INQ_I, ALQ_I, SLQ_I, DUQ_I, CBQ_I, FSQ_I, BPX_I, OHXREF_I, DEMO_I)
for (dataframe in dataframes) {
  df <- merge(df, dataframe, by = "SEQN", all = TRUE)
}
x <- df[, !names(df) %in% c("sum", "SEQN", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050","DPQ060","DPQ070","DPQ080","DPQ090","SDMVPSU","DMDFMSIZ", "DIQ050" )]
x <- x[, !names(x) %in% c("WTMEC2YR", "SDMVSTRA", "SEQN", "depression", "depression_numeric")]
getwd()
write.csv(x,"/Users/Ting/Desktop/Machine learning/data.csv", row.names = TRUE) 

