# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:51:15 2023

@author: cpalmisano
"""

###basic load
#sourcing
import datetime
#import logging
#gen lib
import pandas as pd

import sqlalchemy
from sqlalchemy import text
#####Timing START
import time
from datetime import timedelta
start_time = time.monotonic()


def mooshin(inConn,inJobId, inLogger):      

#####################

    #Connect to DB via SQL Server
    # conn = pyodbc.connect('Driver={SQL Server};'   
    #                       'Server=SERVER;'   ##SERVER NAME
    #                       'Database=DATABASE;'       ##DATABASE
    #                       'Trusted_Connection=yes;')
    # cursor = conn.cursor()
    
    logger = inLogger
    
    conn = inConn
    
    conn.autocommit = True
    
    cursor = conn.cursor()
    
    ###Connect to SQL Server
    con = sqlalchemy.create_engine('mssql://SERVER/DATABASE?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server',
                                    fast_executemany=True)    
    
    #Check for database connection
    if (con == False):
        print("Connection to DATABASE Error")
    else: 
        print("Connection to DATABASE Successful")
    
    #####################################
    
    hfid = """
    Drop Table if exists Temp.HfClaimCheck;
    
    select HfClaimId into Temp.HfClaimCheck from Empire.ClaimsSource_Test where JobId = {0}
            """.format(inJobId)
    
    cursor.execute(hfid)
    
    print("Got past Temp.HfClaimCheck")
    #####Read only Hfclaimids from job into table for non compounding moosh
    # hfid = pd.read_sql_query('''
    # select HfClaimId  from Empire.ClaimsSource_Test where JobId = {0}
    #                              ''', con) 
    
    
    # ### Insert data into SQL Server
    # print('Query Complete. Loading new Moosh Table data to SQL')  
    
    # conn.commit()                  #code to upload dataframe to SQL Server
    # hfid.to_sql( 'HfClaimCheck', con, schema='Temp', index=False, chunksize=100,  if_exists='replace')   
    
    # print("HfClaimIds from job in table")
    
    # #####Timing END
    # end_time = time.monotonic()
    # print('HFClaimIds in table', timedelta(seconds=end_time - start_time))

    ######################################
    logger.info('{0} Starting Moosh from SQL'.format(time.ctime()))
    
    
    moosh = pd.read_sql_query(text('''
    ----Establish 1st table
    
    
    --establish max prices and updating/changing fields
    
    select distinct  a.ClaimNr, a.claimlinenr, a.ClaimAdjustmentNr,  c.[FileName],
    	CASE WHEN c.FileType = 'WGS' OR (len([FileName]) > 82) THEN MAX(a.ClaimChargedAmount) ELSE Sum(a.ClaimChargedAmount) END as Total_ClaimChargedAmount,  --len for CS90 monthly files needing a max not a sum for the amounts
    	CASE WHEN c.FileType = 'WGS' OR (len([FileName]) > 82) THEN MAX(a.ClaimLineChargedAmount) ELSE Sum(a.ClaimLineChargedAmount) END as Total_ClaimLineChargedAmount,  --len for CS90 monthly files needing a max not a sum for the amounts
    	CASE WHEN c.FileType = 'WGS' OR (len([FileName]) > 82) THEN MAX(a.BilledServiceUnitCount) ELSE Sum(a.BilledServiceUnitCount) END as Total_BilledServiceUnitCount,  --len for CS90 monthly files needing a max not a sum for the amounts
    	MAX(a.HfClaimId)  as HfClaimId,
    	MAX(a.ClaimPaidAmount) as Total_ClaimPaidAmount,				
    	MAX(a.ClaimLinePaidAmount) as Total_ClaimLinePaidAmount,		
    	MAX(a.PaidServiceUnitCount) as Total_PaidServiceUnitCount,		
    	MAX(a.CopayAmount) as Total_CopayAmount,						
    	MAX(a.CoinsuranceAmount)	as Total_CoinsuranceAmount,			
    	MAX(a.DeductibleAmount) as Total_DeductibleAmount,				
    	MAX(a.ApprovedAmount)	 as Total_ApprovedAmount,				
    	MAX(a.MemberPenaltyAmount) as Total_MemberPenaltyAmount,		
    	MAX(a.CoveredExpenseAmount) as Total_CoveredExpenseAmount,		
    	MAX(a.AuthorizationNr) as AuthorizationNr,
    	MAX(a.ServiceEndDate) as ServiceEndDate,
    	max(a.JobId) as MostRecentJobID,
    	max(a.patientSSN) as PatientSSN, 
    	Max(a.dependentNr) as DependentNr, 
    	MAX(a.Person_ID) as Person_Id,
    	MAX(a.MemberRelationshipCode) as MemberRelationshipCode,
    	Max(a.RenderingName) as RenderingName, 
    	Max(a.RenderingAddress1) as RenderingAddress1, 
    	Max(a.RenderingAddress2) as RenderingAddress2, 
    	Max(a.RenderingCity) as RenderingCity, 
    	Max(a.RenderingState) as RenderingState, 
    	Max(a.RenderingZipCode) as RenderingZipCode, 
    	Max(a.RenderingZipCode4) as RenderingZipCode4,
    	Max(a.BillingAddress1) as BillingAddress1, 
    	Max(a.BillingAddress2) as BillingAddress2, 
    	Max(a.BillingCity) as BillingCity, 
    	Max(a.BillingState) as BillingState, 
    	Max(a.BillingZipCode) as BillingZipCode, 
    	Max(a.BillingZipCode4) as BillingZipCode4 , 
    	Max(a.EPIN) as EPIN, 
    	Max(a.CountyCode) as CountyCode, 
    	Max(a.PatientDOB) as PatientDOB, 
    	Max(a.HealthCardId) as HealthCardId, 
    	Max(a.RenderingNPI) as RenderingNPI,
    	Max(a.PatientName) as PatientName, 
    	Max(a.PatientGender) as PatientGender, 
    	Max(a.GroupNr) as GroupNr,    ---65
    	Max(a.SubgroupNr) as SubgroupNr, 
    	Max(a.MemberSSN) as MemberSSN, 
    	Max(a.MemberName) as MemberName,
    	Max(a.MemberAddress1) as MemberAddress1,
    	Max(a.MemberAddress2) as MemberAddress2,
    	Max(a.MemberCity) as MemberCity,
    	Max(a.MemberState) as MemberState,
    	Max(a.MemberZipCode) as	MemberZipCode,
    	Max(a.MemberZipCode4) as MemberZipCode4, 
    	 ClaimLineStatusCode, 
    	Max(a.BillingTaxId) as BillingTaxId, 
    	Max(a.ServiceRenderingType) as ServiceRenderingType,
    	Max(a.AdmitDate) as AdmitDate, 
    	Max(a.DischargeDate) as DischargeDate,
    	Max(c.FileType) as FileType,
    	Max(a.PaidDate) as PaidDate, 
    	Max(a.InvestigationClaimCode) as InvestigationClaimCode, 
    	Max(a.HCPCS_Modifier1) as HCPCS_Modifier1, 
    	Max(a.HCPCS_Modifier2) as HCPCS_Modifier2,
    	Max(a.HCPCS_Modifier3) as HCPCS_Modifier3,
    	Max(a.HCPCS_Modifier4) as HCPCS_Modifier4,
    	Max(a.Deceased) as Deceased, 
    	Max(a.BilledServiceUnitCount) as BilledServiceUnitCount, 
    	Max(a.RenderingTaxId) as RenderingTaxId, 
    	Max(a.ClaimChargedAmount) as ClaimChargedAmount, 
    	Max(a.ClaimPaidAmount) as ClaimPaidAmount,    
    	Max(a.DiagnosisCodeAdmit) as DiagnosisCodeAdmit,
    	Max(a.DiagnosisCodeAdmitPOA) as DiagnosisCodeAdmitPOA, 
    	Max(a.DiagnosisCodePrincipal) as DiagnosisCodePrincipal, 
    	Max(a.DiagnosisCodePrincipalPOA) as DiagnosisCodePrincipalPOA, 
    	Max(a.DiagnosisCode1) as DiagnosisCode1, 
    	Max(a.DiagnosisCode1POA) as DiagnosisCode1POA,
    	Max(a.DiagnosisCode2) as DiagnosisCode2, 
    	Max(a.DiagnosisCode2POA) as DiagnosisCode2POA, 
    	Max(a.DiagnosisCode3) as DiagnosisCode3, 
    	Max(a.DiagnosisCode3POA) as DiagnosisCode3POA, 
    	Max(a.DiagnosisCode4) as DiagnosisCode4, 
    	Max(a.DiagnosisCode4POA) as DiagnosisCode4POA, 
    	Max(a.DiagnosisCode5) as DiagnosisCode5, 
    	Max(a.DiagnosisCode5POA) as DiagnosisCode5POA, 
    	Max(a.DiagnosisCode6) as DiagnosisCode6, 
    	Max(a.DiagnosisCode6POA) as DiagnosisCode6POA, 
    	Max(a.DiagnosisCode7) as DiagnosisCode7, 
    	Max(a.DiagnosisCode7POA) as DiagnosisCode7POA, 
    	Max(a.DiagnosisCode8) as DiagnosisCode8, 
    	Max(a.DiagnosisCode8POA) as DiagnosisCode8POA, 
    	Max(a.DiagnosisCode9) as DiagnosisCode9, 
    	Max(a.DiagnosisCode9POA) as DiagnosisCode9POA, 
    	Max(a.DiagnosisCode10) as DiagnosisCode10, 
    	Max(a.DiagnosisCode10POA) as DiagnosisCode10POA, 
    	Max(a.DiagnosisCode11) as DiagnosisCode11, 
    	Max(a.DiagnosisCode11POA) as DiagnosisCode11POA, 
    	Max(a.DiagnosisCode12) as DiagnosisCode12, 
    	Max(a.DiagnosisCode12POA) as DiagnosisCode12POA, 
    	Max(a.DiagnosisCode13) as DiagnosisCode13, 
    	Max(a.DiagnosisCode13POA) as DiagnosisCode13POA, 
    	Max(a.DiagnosisCode14) as DiagnosisCode14, 
    	Max(a.DiagnosisCode14POA) as DiagnosisCode14POA, 
    	Max(a.DiagnosisCode15) as DiagnosisCode15, 
    	Max(a.DiagnosisCode15POA) as DiagnosisCode15POA, 
    	Max(a.DiagnosisCode16) as DiagnosisCode16, 
    	Max(a.DiagnosisCode16POA) as DiagnosisCode16POA, 
    	Max(a.DiagnosisCode17) as DiagnosisCode17, 
    	Max(a.DiagnosisCode17POA) as DiagnosisCode17POA, 
    	Max(a.DiagnosisCode18) as DiagnosisCode18, 
    	Max(a.DiagnosisCode18POA) as DiagnosisCode18POA, 
    	Max(a.DiagnosisCode19) as DiagnosisCode19, 
    	Max(a.DiagnosisCode19POA) as DiagnosisCode19POA, 
    	Max(a.DiagnosisCode20) as DiagnosisCode20, 
    	Max(a.DiagnosisCode20POA) as DiagnosisCode20POA, 
    	Max(a.DiagnosisCode21) as DiagnosisCode21, 
    	Max(a.DiagnosisCode21POA) as DiagnosisCode21POA, 
    	Max(a.DiagnosisCode22) as DiagnosisCode22, 
    	Max(a.DiagnosisCode22POA) as DiagnosisCode22POA, 
    	Max(a.DiagnosisCode23) as DiagnosisCode23, 
    	Max(a.DiagnosisCode23POA) as DiagnosisCode23POA,  
    	Max(a.DiagnosisCode24) as DiagnosisCode24, 
    	Max(a.DiagnosisCode24POA) as DiagnosisCode24POA, 
    	Max(a.DiagnosisCode25) as DiagnosisCode25, 
    	Max(a.DiagnosisCode25POA) as DiagnosisCode25POA, 
    	Max(a.ProcedureCodeSurgical) as ProcedureCodeSurgical, 
    	Max(a.ProcedureCode1) as ProcedureCode1,
    	Max(a.ProcedureCode2) as ProcedureCode2,
    	Max(a.ProcedureCode3) as ProcedureCode3,
    	Max(a.ProcedureCode4) as ProcedureCode4,
    	Max(a.ProcedureCode5) as ProcedureCode5,
    	Max(a.ProcedureCode6) as ProcedureCode6,
    	Max(a.ProcedureCode7) as ProcedureCode7,
    	Max(a.ProcedureCode8) as ProcedureCode8,
    	Max(a.ProcedureCode9) as ProcedureCode9,
    	Max(a.ProcedureCode10) as ProcedureCode10,
    	Max(a.ProcedureCode11) as ProcedureCode11,
    	Max(a.ProcedureCode12) as ProcedureCode12,
    	Max(a.ProcedureCode13) as ProcedureCode13,
    	Max(a.ProcedureCode14) as ProcedureCode14,
    	Max(a.ProcedureCode15) as ProcedureCode15,
    	Max(a.ProcedureCode16) as ProcedureCode16,
    	Max(a.ProcedureCode17) as ProcedureCode17,
    	Max(a.ProcedureCode18) as ProcedureCode18,
    	Max(a.ProcedureCode19) as ProcedureCode19,
    	Max(a.ProcedureCode20) as ProcedureCode20,
    	Max(a.ProcedureCode21) as ProcedureCode21,
    	Max(a.ProcedureCode22) as ProcedureCode22,
    	Max(a.ProcedureCode23) as ProcedureCode23,
    	Max(a.ProcedureCode24) as ProcedureCode24,
    	Max(a.ProcedureCode25) as ProcedureCode25, 
    	Max(a.ProfitabilityCode) as ProfitabilityCode, 
    	Max(a.Region) as Region, 
    	Max(a.ServiceStartDate) as ServiceStartDate, 
    	Max(a.PlaceOfService) as PlaceOfService, 
    	Max(a.ClaimLineChargedAmount) as ClaimLineChargedAmount, 
    	Max(a.ClaimLinePaidAmount) as ClaimLinePaidAmount, 
    	Max(a.ProviderLocationCode) as ProviderLocationCode, 
    	Max(a.InNetworkCode) as InNetworkCode, 
    	Max(a.ProviderClassCode) as ProviderClassCode, 
    	Max(a.Par) as Par, 
    	Max(a.PaidServiceUnitCount) as PaidServiceUnitCount, 
    	Max(a.CopayAmount) as CopayAmount, 
    	Max(a.CoinsuranceAmount) as CoinsuranceAmount, 
    	Max(a.DeductibleAmount) as DeductibleAmount, 
    	Max(a.ApprovedAmount) as ApprovedAmount,
    	Max(a.MemberPenaltyAmount) as MemberPenaltyAmount, 
    	Max(a.CoveredExpenseAmount) as CoveredExpenseAmount, 
    	Max(a.ClaimEntryDate) as ClaimEntryDate,    
    	Max(a.PrimaryCarrierResponsibilityCode) as PrimaryCarrierResponsibilityCode,
    	Max(a.ProcesserUnitId) as ProcesserUnitId, 
    	Max(a.PackageNr) as PackageNr, 
    	Max(a.EmployerGroupDepartmentNr) as EmployerGroupDepartmentNr, 
    	Max(a.COBSavingsAmount) as COBSavingsAmount, 
    	MAX(a.BillingNPI) as BillingNPI, 
    	 IsSingleContract,
    	Max(a.ClaimsSourceSortId) as ID,
    	Max(a.DischargeStatus) as DischargeStatus,
    	Max(a.AdmitTypeCode) as AdmitTypeCode,
    	Max(a.DiagnosisRelatedGroup) as DiagnosisRelatedGroup, 
    	Max(a.ICDVersion) as ICDVersion,
    	Max(a.TypeOfBillCode) as TypeOfBillCode,
    	Max(a.DiagnosisRelatedGroupType) as DiagnosisRelatedGroupType,
    	Max(a.DiagnosisRelatedGroupSeverity) as DiagnosisRelatedGroupSeverity,
    	Max(a.RateCategory) as RateCategory,
    	Max(a.RateSubcategory) as RateSubcategory,
    	Max(a.RevenueCode) as RevenueCode,
    	Max(a.BenefitPaymentTierCode) as BenefitPaymentTierCode,
    	Max(a.PostDate) as PostDate,
        Max(a.PreferredIndicator) as PreferredIndicator,
    	MAX(a.ValueFunctionCode1) as ValueFunctionCode1,
    	MAX(a.ValueFunctionCode2) as ValueFunctionCode2,
    	MAX(a.ValueFunctionCode3) as ValueFunctionCode3 
    	, DenialReasonCode
        , denialReasonDescription
        , PaidDate
        , PostDate 
        , benefitcategorycode
    	, benefitcategorycodedescription
    	, surprisebilling
    	, qpaamount
    	, BillingName
    	, HCPCS
    	, ProviderSpecialtyCode
    	, ProviderContractType
    	, MemberAddress1
    	, MemberAddress2
    	, MemberCity
    	, MemberState
    	, MemberZipCode
    	, MemberZipCode4
    	, RenderingName
    	, RenderingAddress1
    	, RenderingAddress2
    	, RenderingCity
    	, RenderingState
    	, RenderingZipCode
    	, RenderingZipCode4
    	, BillingAddress1
    	, BillingAddress2
    	, BillingCity
    	, BillingState
    	, BillingZipCode
    	 from Empire.ClaimsSource_Test a
             LEFT JOIN Main.Jobs c
                       on a.JobId = c.jobid
              WHERE EXISTS (Select distinct HfClaimId from Temp.HfClaimCheck cc
								where a.HfClaimId = cc.HfClaimId)
          		GROUP BY a.ClaimNr, a.claimlinenr, a.ClaimAdjustmentNr, FileType, ClaimLineStatusCode, IsSingleContract, [FileName]  ,
                    denialReasonCode, denialReasonDescription, PaidDate, PostDate,  benefitcategorycode	, benefitcategorycodedescription
            		, surprisebilling, qpaamount, BillingName, HCPCS, ProviderSpecialtyCode, ProviderContractType, MemberAddress1
            		, MemberAddress2, MemberCity, MemberState, MemberZipCode, MemberZipCode4, RenderingName, RenderingAddress1
            		, RenderingAddress2, RenderingCity, RenderingState, RenderingZipCode, RenderingZipCode4, BillingAddress1
            		, BillingAddress2, BillingCity, BillingState, BillingZipCode
            ''' ), con)
            
            ##### 
    ### Insert data into SQL Server
    logger.info('{0} Query Complete. Loading new Moosh Table data to SQL'.format(time.ctime()))  
    
    conn.commit()                  #code to upload dataframe to SQL Server
    moosh.to_sql( 'ClaimsMooshA', con, schema='Temp', index=False, chunksize=5000,  if_exists='replace')   
            
    logger.info('{0} New data loaded into Moosh table'.format(time.ctime()))        
            
    #####Timing END
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    ###############################################################################################################
    
    print('Starting flag Moosh')
    flagmoosh = pd.read_sql_query(text('''                        
    --set flags for prices if canceled out and add into previous table of max prices and int fields
    
    select distinct  a.ClaimNr,	a.ClaimAdjustmentNr, a.claimlinenr,
    						MAX(b.FLAG_ClaimChargedAmt) AS CancelOut_ClaimChargedAmt ,
    						MAX(b.FLAG_ClaimPaidAmt) AS CancelOut_ClaimPaidAmt ,
    						MAX(b.FLAG_ClaimLineChargedAmt) AS CancelOut_ClaimLineChargedAmt ,
    						MAX(b.FLAG_ClaimLinePaidAmt) AS CancelOut_ClaimLinePaidAmt ,
    						MAX(b.FLAG_PaidServiceUnitCnt) AS CancelOut_PaidServiceUnitCnt ,
    						MAX(b.FLAG_CopayAmt) AS CancelOut_CopayAmt ,
    						MAX(b.FLAG_CoinsuranceAmt) AS CancelOut_CoinsuranceAmt ,
    						MAX(b.FLAG_DeductibleAmt) AS CancelOut_DeductibleAmt,
    						MAX(b.FLAG_ApprovedAmt) AS CancelOut_ApprovedAmt ,
    						MAX(b.FLAG_MemberPenaltyAmt) AS CancelOut_MemberPenaltyAmt ,
    						MAX(b.FLAG_CoveredExpenseAmt) AS CancelOut_CoveredExpenseAmt ,
    						MAX(b.FLAG_BilledServiceUnitCnt) AS CancelOut_BilledServiceUnitCnt ,
                            MAX(Total_ClaimPaidAmount) AS Total_ClaimPaidAmount,
                            MAX(Total_ClaimLinePaidAmount) AS Total_ClaimLinePaidAmount ,
                            MAX(Total_PaidServiceUnitCount ) AS Total_PaidServiceUnitCount ,
                            MAX(Total_CopayAmount ) AS Total_CopayAmount,
                            MAX(Total_CoinsuranceAmount ) AS Total_CoinsuranceAmount,
                            MAX(Total_DeductibleAmount) AS Total_DeductibleAmount,
                            MAX( Total_ApprovedAmount) AS  Total_ApprovedAmount,
                            MAX( Total_MemberPenaltyAmount) AS Total_MemberPenaltyAmount,
                            MAX(Total_CoveredExpenseAmount) as Total_CoveredExpenseAmount,
                            MAX(Total_BilledServiceUnitCount) as Total_BilledServiceUnitCount
    				    --into #temp1a
    				from Temp.ClaimsMooshA a 
     				 LEFT JOIN (
     					SELECT  ClaimNr, ClaimAdjustmentNr,
    						CASE WHEN SUM(TOTAL_CLAIMCHARGEDAMOUNT)=0 THEN '1' ELSE '0' END AS FLAG_ClaimChargedAmt,
    						CASE WHEN SUM(Total_ClaimPaidAmount)=0 THEN '1' ELSE '0' END AS FLAG_ClaimPaidAmt,
    						CASE WHEN SUM(Total_ClaimLineChargedAmount)=0 THEN '1' ELSE '0' END AS FLAG_ClaimLineChargedAmt,
    						CASE WHEN SUM(Total_ClaimLinePaidAmount)=0 THEN '1' ELSE '0' END AS FLAG_ClaimLinePaidAmt,
    						CASE WHEN SUM(Total_PaidServiceUnitCount)=0 THEN '1' ELSE '0' END AS FLAG_PaidServiceUnitCnt,
    						CASE WHEN SUM(Total_CopayAmount)=0 THEN '1' ELSE '0' END AS FLAG_CopayAmt,
    						CASE WHEN SUM(Total_CoinsuranceAmount)=0 THEN '1' ELSE '0' END AS FLAG_CoinsuranceAmt,
    						CASE WHEN SUM(Total_DeductibleAmount)=0 THEN '1' ELSE '0' END AS FLAG_DeductibleAmt,
    						CASE WHEN SUM(Total_ApprovedAmount)=0 THEN '1' ELSE '0' END AS FLAG_ApprovedAmt,
    						CASE WHEN SUM(Total_MemberPenaltyAmount)=0 THEN '1' ELSE '0' END AS FLAG_MemberPenaltyAmt,
    						CASE WHEN SUM(Total_CoveredExpenseAmount)=0 THEN '1' ELSE '0' END AS FLAG_CoveredExpenseAmt,
    						CASE WHEN SUM(Total_BilledServiceUnitCount)=0 THEN '1' ELSE '0' END AS FLAG_BilledServiceUnitCnt
     					  FROM Temp.ClaimsMooshA
    								GROUP BY CLAIMNR, CLAIMADJUSTMENTNR
     							)  b
     					ON a.claimnr = b.claimnr AND a.ClaimAdjustmentNr = b.ClaimAdjustmentNr
    				group by a.ClaimNr,	a.claimlinenr,	a.ClaimAdjustmentNr
    	
          ''') , con)
          
    ### Insert data into SQL Server

    logger.info('{0} Loading new MooshFlag table data to SQL'.format(time.ctime()))  
    conn.commit()                  #code to upload dataframe to SQL Server
    flagmoosh.to_sql( 'ClaimsMooshFlag', con, schema='Temp', index=False, chunksize=5000,  if_exists='replace')   
    
    logger.info('{0} New data loaded into MooshFlag table'.format(time.ctime()))  
    
    #####Timing END
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    
    ####################################################################################################################
    
    print("Starting CTE Moosh ")
    ctemoosh = pd.read_sql_query(text('''
    ---------------------------Fixes problems with NULLS and problem columns
    			
    ---establishes last non null values WITH NULLS 
    with cte_grp AS 
    				  (
    					SELECT 
    						claimnr
                            , ClaimAdjustmentNr
    						, claimlinenr
    						, DenialReasonCode
    						, DenialReasonDescription
    						, PaidDate
    						, PostDate 
    						, Total_ClaimChargedAmount
    						, Total_ClaimLineChargedAmount
    						, Total_BilledServiceUnitCount
    						, benefitcategorycode
    						, benefitcategorycodedescription
    						, surprisebilling
    						, qpaamount
    						, BillingName
    						, HCPCS
    						, ProviderSpecialtyCode
    						, ProviderContractType
    						, MemberAddress1
    						, MemberAddress2
    						, MemberCity
    						, MemberState
    						, MemberZipCode
    						, MemberZipCode4
    						, MostRecentJobID 
    						, RenderingName
    						, RenderingAddress1
    						, RenderingAddress2
    						, RenderingCity
    						, RenderingState
    						, RenderingZipCode
    						, RenderingZipCode4
    						, BillingAddress1
    						, BillingAddress2
    						, BillingCity
    						, BillingState
    						, BillingZipCode	,
    						grp  =   MAX(IIF(DenialReasonCode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr ROWS Unbounded PRECEDING),
    						grp1 =  MAX(IIF(DenialReasonDescription IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp2 =  MAX(IIF(a.PaidDate IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr ROWS Unbounded PRECEDING),
    						grp3 =  MAX(IIF(a.PostDate IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr ROWS Unbounded PRECEDING),
    						grp5 =  MAX(IIF(Total_ClaimChargedAmount  IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr ROWS Unbounded PRECEDING),
    						grp6 =  MAX(IIF(Total_ClaimLineChargedAmount  IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp7 =  MAX(IIF(Total_BilledServiceUnitCount IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp8 =  MAX(IIF(benefitcategorycode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp9 =  MAX(IIF(benefitcategorycodedescription IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp10 = MAX(IIF(surprisebilling IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp11 = MAX(IIF(qpaamount IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp12 = MAX(IIF(BillingName IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp13 = MAX(IIF(HCPCS IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp14 = MAX(IIF(ProviderSpecialtyCode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp15 = MAX(IIF(ProviderContractType IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp16 = MAX(IIF(MemberAddress1 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp17 = MAX(IIF(MemberAddress2 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp18 = MAX(IIF(MemberCity IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp19 = MAX(IIF(MemberState IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp20 = MAX(IIF(MemberZipCode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp21 = MAX(IIF(MemberZipCode4 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp22 = MAX(IIF(MostRecentJobID IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp23 = MAX(IIF(RenderingName IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp24 = MAX(IIF(RenderingAddress1 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp25 = MAX(IIF(RenderingAddress2 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp26 = MAX(IIF(RenderingCity IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp27 = MAX(IIF(RenderingState IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp28 = MAX(IIF(RenderingZipCode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp29= MAX(IIF(RenderingZipCode4 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp30 = MAX(IIF(BillingAddress1 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp31 = MAX(IIF(BillingAddress2 IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp32 = MAX(IIF(BillingCity IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp33 = MAX(IIF(BillingState IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						grp34 = MAX(IIF(BillingZipCode IS NOT NULL, a.ID , NULL)) OVER (PARTITION BY a.claimnr, a.ClaimAdjustmentNr, a.claimlinenr ORDER by  a.ClaimAdjustmentNr, a.claimlinenr  ROWS Unbounded PRECEDING),
    						a.ID
    						from Temp.ClaimsMooshA a 	
    					)										
    				Select 
    					distinct claimnr, claimlinenr, ClaimAdjustmentNr	
    					, DenialReasonCode =				 MAX(DenialReasonCode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING)
    					, DenialReasonDescription =			 MAX(DenialReasonDescription) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp1 ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING)
    					, PaidDate =						 MAX(PaidDate) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp2 ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING)
    					, PostDate =						 MAX(PostDate) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp3 ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING) 
    					, Total_ClaimChargedAmount =		 MAX(Total_ClaimChargedAmount) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp5 ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING) 
    					, Total_ClaimLineChargedAmount =	 MAX(Total_ClaimLineChargedAmount) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp6 ORDER by  ClaimAdjustmentNr, claimlinenr  ROWS Unbounded PRECEDING) 
    					, Total_BilledServiceUnitCount =	 MAX(Total_BilledServiceUnitCount) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp7 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, benefitcategorycode =				 MAX(benefitcategorycode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp8 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, benefitcategorycodedescription =	 MAX(benefitcategorycodedescription) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp9 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, surprisebilling =					 MAX(surprisebilling) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp10 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, qpaamount =						 MAX(qpaamount) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp11 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, BillingName =						 MAX(BillingName) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp12 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, HCPCS =							 MAX(HCPCS) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp13 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, ProviderSpecialtyCode =			 MAX(ProviderSpecialtyCode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp14 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, ProviderContractType =			 MAX(ProviderContractType) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp15 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, MemberAddress1 =					 MAX(MemberAddress1) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp16 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, MemberAddress2 =					 MAX(MemberAddress2) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp17 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, MemberCity =						 MAX(MemberCity) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp18 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, MemberState =						 MAX(MemberState) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp19 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, MemberZipCode =					 MAX(MemberZipCode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp20 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, MemberZipCode4 =					 MAX(MemberZipCode4) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp21 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, MostRecentJobID =					 MAX(MostRecentJobID) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp22 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, RenderingName =					 MAX(RenderingName) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp23 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, RenderingAddress1 =				 MAX(RenderingAddress1) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp24 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, RenderingAddress2 =				 MAX(RenderingAddress2) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp25 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, RenderingCity =					 MAX(RenderingCity) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp26 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, RenderingState =					 MAX(RenderingState) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp27 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, RenderingZipCode =				 MAX(RenderingZipCode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp28 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, RenderingZipCode4 =				 MAX(RenderingZipCode4) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp29 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, BillingAddress1 =					 MAX(BillingAddress1) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp30 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, BillingAddress2 =					 MAX(BillingAddress2) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp31 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, BillingCity =						 MAX(BillingCity) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp32 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING) 
    					, BillingState =					 MAX(BillingState) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp33 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, BillingZipCode =					 MAX(BillingZipCode) OVER (PARTITION BY claimnr, ClaimAdjustmentNr, claimlinenr, grp34 ORDER by  ClaimAdjustmentNr, claimlinenr ROWS Unbounded PRECEDING)
    					, ID
    						---into #ctemp
    					FROM cte_grp
    						order by claimnr,ClaimAdjustmentNr,claimlinenr; 
    
          ''') , con)
          
          
    ### Insert data into SQL Server
    logger.info('{0} Query Complete. Loading new Moosh CTE table data to SQL'.format(time.ctime()))  
    conn.commit()                  #code to upload dataframe to SQL Server
    ctemoosh.to_sql( 'ClaimsMooshCte', con, schema='Temp', index=False, chunksize=5000,  if_exists='replace')   
           
    logger.info('{0} New data loaded into Moosh CTE table'.format(time.ctime()))      
    
    #####Timing END
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    ##############################################################################################################################			
    
    
    print('Starting Set Moosh')
    claimsset = pd.read_sql_query(text('''        
    				select distinct a.ClaimNr
    				,MAX(a.ClaimAdjustmentNr) AS ClaimAdjustmentNr
    				, a.claimlinenr
    				, a.HfClaimId  
    				,MAX(c.Total_ClaimPaidAmount) AS Total_ClaimPaidAmount
    				,MAX(c.Total_ClaimLinePaidAmount) AS Total_ClaimLinePaidAmount
    				,MAX(c.Total_PaidServiceUnitCount) AS Total_PaidServiceUnitCount
    				,MAX(c.Total_CopayAmount) AS Total_CopayAmount
    				,MAX(c.Total_CoinsuranceAmount) AS Total_CoinsuranceAmount
    				,MAX(c.Total_DeductibleAmount) AS Total_DeductibleAmount
    				,MAX(c.Total_ApprovedAmount) AS Total_ApprovedAmount
    				,MAX(c.Total_MemberPenaltyAmount) AS Total_MemberPenaltyAmount
    				,MAX(c.Total_CoveredExpenseAmount) AS Total_CoveredExpenseAmount
    				,MAX(a.AuthorizationNr) AS AuthorizationNr
    				,MAX(a.ServiceEndDate) AS ServiceEndDate
    				,MAX(b.MostRecentJobID) AS MostRecentJobID
    				,MAX(a.PatientSSN) AS PatientSSN
    				,MAX(a.DependentNr) AS DependentNr
    				,MAX(a.Person_Id) AS Person_Id
    				,MAX(a.MemberRelationshipCode) AS MemberRelationshipCode
    				,MAX(b.RenderingName) AS RenderingName
    				,MAX(b.RenderingAddress1) AS RenderingAddress1
    				,MAX(b.RenderingAddress2) AS RenderingAddress2
    				,MAX(b.RenderingCity) AS RenderingCity
    				,MAX(b.RenderingState) AS RenderingState
    				,MAX(b.RenderingZipCode) AS RenderingZipCode
    				,MAX(b.RenderingZipCode4) AS RenderingZipCode4
    				,MAX(b.BillingAddress1) AS BillingAddress1
    				,MAX(b.BillingAddress2) AS BillingAddress2
    				,MAX(b.BillingCity) AS BillingCity
    				,MAX(b.BillingState) AS BillingState
    				,MAX(b.BillingZipCode) AS BillingZipCode
    				,MAX(a.BillingZipCode4) AS BillingZipCode4
    				,MAX(a.EPIN) AS EPIN
    				,MAX(a.CountyCode) AS CountyCode
    				,MAX(a.PatientDOB) AS PatientDOB
    				,MAX(a.HealthCardId) AS HealthCardId
    				,MAX(a.RenderingNPI) AS RenderingNPI
    				,MAX(a.PatientName) AS PatientName
    				,MAX(a.PatientGender) AS PatientGender
    				,MAX(a.GroupNr) AS GroupNr
    				,MAX(a.SubgroupNr) AS SubgroupNr
    				,MAX(a.MemberSSN) AS MemberSSN
    				,MAX(a.MemberName) AS MemberName
    				, a.ClaimLineStatusCode
    				,MAX(a.BillingTaxId) AS BillingTaxId
    				,MAX(a.ServiceRenderingType) AS ServiceRenderingType
    				,MAX(a.AdmitDate) AS AdmitDate
    				,MAX(a.DischargeDate) AS DischargeDate
    				,MAX(a.FileType) AS FileType
    				,MAX(a.InvestigationClaimCode) AS InvestigationClaimCode
    				,MAX(a.HCPCS_Modifier1) AS HCPCS_Modifier1
    				,MAX(a.HCPCS_Modifier2) AS HCPCS_Modifier2
    				,MAX(a.HCPCS_Modifier3) AS HCPCS_Modifier3
    				,MAX(a.HCPCS_Modifier4) AS HCPCS_Modifier4
    				,MAX(a.Deceased) AS Deceased
    				,MAX(a.BilledServiceUnitCount) AS BilledServiceUnitCount
    				,MAX(a.RenderingTaxId) AS RenderingTaxId
    				,MAX(a.ClaimChargedAmount) AS ClaimChargedAmount
    				,MAX(a.ClaimPaidAmount) AS ClaimPaidAmount
    				,MAX(a.DiagnosisCodeAdmit) AS DiagnosisCodeAdmit
    				,MAX(a.DiagnosisCodeAdmitPOA) AS DiagnosisCodeAdmitPOA
    				,MAX(a.DiagnosisCodePrincipal) AS DiagnosisCodePrincipal
    				,MAX(a.DiagnosisCodePrincipalPOA) AS DiagnosisCodePrincipalPOA
    				,MAX(a.DiagnosisCode1) AS DiagnosisCode1
    				,MAX(a.DiagnosisCode1POA) AS DiagnosisCode1POA
    				,MAX(a.DiagnosisCode2) AS DiagnosisCode2
    				,MAX(a.DiagnosisCode2POA) AS DiagnosisCode2POA
    				,MAX(a.DiagnosisCode3) AS DiagnosisCode3
    				,MAX(a.DiagnosisCode3POA) AS DiagnosisCode3POA
    				,MAX(a.DiagnosisCode4) AS DiagnosisCode4
    				,MAX(a.DiagnosisCode4POA) AS DiagnosisCode4POA
    				,MAX(a.DiagnosisCode5) AS DiagnosisCode5
    				,MAX(a.DiagnosisCode5POA) AS DiagnosisCode5POA
    				,MAX(a.DiagnosisCode6) AS DiagnosisCode6
    				,MAX(a.DiagnosisCode6POA) AS DiagnosisCode6POA
    				,MAX(a.DiagnosisCode7) AS DiagnosisCode7
    				,MAX(a.DiagnosisCode7POA) AS DiagnosisCode7POA
    				,MAX(a.DiagnosisCode8) AS DiagnosisCode8
    				,MAX(a.DiagnosisCode8POA) AS DiagnosisCode8POA
    				,MAX(a.DiagnosisCode9) AS DiagnosisCode9
    				,MAX(a.DiagnosisCode9POA) AS DiagnosisCode9POA
    				,MAX(a.DiagnosisCode10) AS DiagnosisCode10
    				,MAX(a.DiagnosisCode10POA) AS DiagnosisCode10POA
    				,MAX(a.DiagnosisCode11) AS DiagnosisCode11
    				,MAX(a.DiagnosisCode11POA) AS DiagnosisCode11POA
    				,MAX(a.DiagnosisCode12) AS DiagnosisCode12
    				,MAX(a.DiagnosisCode12POA) AS DiagnosisCode12POA
    				,MAX(a.DiagnosisCode13) AS DiagnosisCode13
    				,MAX(a.DiagnosisCode13POA) AS DiagnosisCode13POA
    				,MAX(a.DiagnosisCode14) AS DiagnosisCode14
    				,MAX(a.DiagnosisCode14POA) AS DiagnosisCode14POA
    				,MAX(a.DiagnosisCode15) AS DiagnosisCode15
    				,MAX(a.DiagnosisCode15POA) AS DiagnosisCode15POA
    				,MAX(a.DiagnosisCode16) AS DiagnosisCode16
    				,MAX(a.DiagnosisCode16POA) AS DiagnosisCode16POA
    				,MAX(a.DiagnosisCode17) AS DiagnosisCode17
    				,MAX(a.DiagnosisCode17POA) AS DiagnosisCode17POA
    				,MAX(a.DiagnosisCode18) AS DiagnosisCode18
    				,MAX(a.DiagnosisCode18POA) AS DiagnosisCode18POA
    				,MAX(a.DiagnosisCode19) AS DiagnosisCode19
    				,MAX(a.DiagnosisCode19POA) AS DiagnosisCode19POA
    				,MAX(a.DiagnosisCode20) AS DiagnosisCode20
    				,MAX(a.DiagnosisCode20POA) AS DiagnosisCode20POA
    				,MAX(a.DiagnosisCode21) AS DiagnosisCode21
    				,MAX(a.DiagnosisCode21POA) AS DiagnosisCode21POA
    				,MAX(a.DiagnosisCode22) AS DiagnosisCode22
    				,MAX(a.DiagnosisCode22POA) AS DiagnosisCode22POA
    				,MAX(a.DiagnosisCode23) AS DiagnosisCode23
    				,MAX(a.DiagnosisCode23POA) AS DiagnosisCode23POA
    				,MAX(a.DiagnosisCode24) AS DiagnosisCode24
    				,MAX(a.DiagnosisCode24POA) AS DiagnosisCode24POA
    				,MAX(a.DiagnosisCode25) AS DiagnosisCode25
    				,MAX(a.DiagnosisCode25POA) AS DiagnosisCode25POA
    				,MAX(a.ProcedureCodeSurgical) AS ProcedureCodeSurgical
    				,MAX(a.ProcedureCode1) AS ProcedureCode1
    				,MAX(a.ProcedureCode2) AS ProcedureCode2
    				,MAX(a.ProcedureCode3) AS ProcedureCode3
    				,MAX(a.ProcedureCode4) AS ProcedureCode4
    				,MAX(a.ProcedureCode5) AS ProcedureCode5
    				,MAX(a.ProcedureCode6) AS ProcedureCode6
    				,MAX(a.ProcedureCode7) AS ProcedureCode7
    				,MAX(a.ProcedureCode8) AS ProcedureCode8
    				,MAX(a.ProcedureCode9) AS ProcedureCode9
    				,MAX(a.ProcedureCode10) AS ProcedureCode10
    				,MAX(a.ProcedureCode11) AS ProcedureCode11
    				,MAX(a.ProcedureCode12) AS ProcedureCode12
    				,MAX(a.ProcedureCode13) AS ProcedureCode13
    				,MAX(a.ProcedureCode14) AS ProcedureCode14
    				,MAX(a.ProcedureCode15) AS ProcedureCode15
    				,MAX(a.ProcedureCode16) AS ProcedureCode16
    				,MAX(a.ProcedureCode17) AS ProcedureCode17
    				,MAX(a.ProcedureCode18) AS ProcedureCode18
    				,MAX(a.ProcedureCode19) AS ProcedureCode19
    				,MAX(a.ProcedureCode20) AS ProcedureCode20
    				,MAX(a.ProcedureCode21) AS ProcedureCode21
    				,MAX(a.ProcedureCode22) AS ProcedureCode22
    				,MAX(a.ProcedureCode23) AS ProcedureCode23
    				,MAX(a.ProcedureCode24) AS ProcedureCode24
    				,MAX(a.ProcedureCode25) AS ProcedureCode25
    				,MAX(a.ProfitabilityCode) AS ProfitabilityCode
    				,MAX(a.Region) AS Region
    				,MAX(a.ServiceStartDate) AS ServiceStartDate
    				,MAX(a.PlaceOfService) AS PlaceOfService
    				,MAX(a.ClaimLineChargedAmount) AS ClaimLineChargedAmount
    				,MAX(a.ClaimLinePaidAmount) AS ClaimLinePaidAmount
    				,MAX(a.ProviderLocationCode) AS ProviderLocationCode
    				,MAX(a.InNetworkCode) AS InNetworkCode
    				,MAX(a.ProviderClassCode) AS ProviderClassCode
    				,MAX(a.Par) AS Par
    				,MAX(a.PaidServiceUnitCount) AS PaidServiceUnitCount
    				,MAX(a.CopayAmount) AS CopayAmount
    				,MAX(a.CoinsuranceAmount) AS CoinsuranceAmount
    				,MAX(a.DeductibleAmount) AS DeductibleAmount
    				,MAX(a.ApprovedAmount) AS ApprovedAmount
    				,MAX(a.MemberPenaltyAmount) AS MemberPenaltyAmount
    				,MAX(a.CoveredExpenseAmount) AS CoveredExpenseAmount
    				,MAX(a.ClaimEntryDate) AS ClaimEntryDate
    				,MAX(a.PrimaryCarrierResponsibilityCode) AS PrimaryCarrierResponsibilityCode
    				,MAX(a.ProcesserUnitId) AS ProcesserUnitId
    				,MAX(a.PackageNr) AS PackageNr
    				,MAX(a.EmployerGroupDepartmentNr) AS EmployerGroupDepartmentNr
    				,MAX(a.COBSavingsAmount) AS COBSavingsAmount
    				,MAX(a.BillingNPI) AS BillingNPI
    				, a.IsSingleContract
    				,MAX(a.DischargeStatus) AS DischargeStatus
    				,MAX(a.AdmitTypeCode) AS AdmitTypeCode
    				,MAX(a.DiagnosisRelatedGroup) AS DiagnosisRelatedGroup
    				,MAX(a.ICDVersion) AS ICDVersion
    				,MAX(a.TypeOfBillCode) AS TypeOfBillCode
    				,MAX(a.DiagnosisRelatedGroupType) AS DiagnosisRelatedGroupType
    				,MAX(a.DiagnosisRelatedGroupSeverity) AS DiagnosisRelatedGroupSeverity
    				,MAX(a.RateCategory) AS RateCategory
    				,MAX(a.RateSubcategory) AS RateSubcategory
    				,MAX(a.RevenueCode) AS RevenueCode
    				,MAX(a.BenefitPaymentTierCode) AS BenefitPaymentTierCode
    				,MAX(a.PreferredIndicator) AS PreferredIndicator
    				,MAX(a.ValueFunctionCode1) AS ValueFunctionCode1
    				,MAX(a.ValueFunctionCode2) AS ValueFunctionCode2
    				,MAX(a.ValueFunctionCode3) AS ValueFunctionCode3
    				,MAX(c.CancelOut_ClaimChargedAmt) AS CancelOut_ClaimChargedAmt
    				,MAX(c.CancelOut_ClaimPaidAmt) AS CancelOut_ClaimPaidAmt
    				,MAX(c.CancelOut_ClaimLineChargedAmt) AS CancelOut_ClaimLineChargedAmt
    				,MAX(c.CancelOut_ClaimLinePaidAmt) AS CancelOut_ClaimLinePaidAmt
    				,MAX(c.CancelOut_PaidServiceUnitCnt) AS CancelOut_PaidServiceUnitCnt
    				,MAX(c.CancelOut_CopayAmt) AS CancelOut_CopayAmt
    				,MAX(c.CancelOut_CoinsuranceAmt) AS CancelOut_CoinsuranceAmt
    				,MAX(c.CancelOut_DeductibleAmt) AS CancelOut_DeductibleAmt
    				,MAX(c.CancelOut_ApprovedAmt) AS CancelOut_ApprovedAmt
    				,MAX(c.CancelOut_MemberPenaltyAmt) AS CancelOut_MemberPenaltyAmt
    				,MAX(c.CancelOut_CoveredExpenseAmt) AS CancelOut_CoveredExpenseAmt
    				,MAX(c.CancelOut_BilledServiceUnitCnt) AS CancelOut_BilledServiceUnitCnt
    				,MAX(b.DenialReasonCode) AS DenialReasonCode
    				,MAX(b.DenialReasonDescription) AS DenialReasonDescription
    				,MAX(b.PaidDate) AS PaidDate
    				,MAX(b.PostDate) AS PostDate
    				,MAX(b.Total_ClaimChargedAmount) AS Total_ClaimChargedAmount
    				,MAX(b.Total_ClaimLineChargedAmount) AS Total_ClaimLineChargedAmount
    				,MAX(b.Total_BilledServiceUnitCount) AS Total_BilledServiceUnitCount
    				,MAX(b.benefitcategorycode) AS benefitcategorycode
    				,MAX(b.benefitcategorycodedescription) AS benefitcategorycodedescription
    				,MAX(b.surprisebilling) AS surprisebilling
    				,MAX(b.qpaamount) AS qpaamount
    				,MAX(b.BillingName) AS BillingName
    				,MAX(b.HCPCS) AS HCPCS
    				,MAX(b.ProviderSpecialtyCode) AS ProviderSpecialtyCode
    				,MAX(b.ProviderContractType) AS ProviderContractType
    				,MAX(b.MemberAddress1) AS MemberAddress1
    				,MAX(b.MemberAddress2) AS MemberAddress2
    				,MAX(b.MemberCity) AS MemberCity
    				,MAX(b.MemberState) AS MemberState
    				,MAX(b.MemberZipCode) AS MemberZipCode
    				,MAX(b.MemberZipCode4) AS MemberZipCode4					
    					FROM Temp.ClaimsMooshA a
    					  LEFT JOIN  Temp.ClaimsMooshCte b 
    						ON b.ClaimNr = a.ClaimNr 
    						 AND b.ClaimAdjustmentNr = a.ClaimAdjustmentNr
    						 AND b.ClaimlineNr = a.ClaimlineNr 
    					  LEFT JOIN Temp.ClaimsMooshFlag c
    						 ON a.ClaimNr = c.ClaimNr 
    						 AND a.ClaimAdjustmentNr = c.ClaimAdjustmentNr
    						 AND a.ClaimlineNr = c.ClaimlineNr 
    							group by a.ClaimNr, a.ClaimAdjustmentNr, a.ClaimlineNr, a.HfClaimId, a.ClaimLineStatusCode, a.IsSingleContract
                                  '''), con)
        
    ### Insert data into SQL Server

    logger.info('{0} Query Complete. Loading new Claims Set table data to SQL'.format(time.ctime()))     
    conn.commit()                  #code to upload dataframe to SQL Server
    claimsset.to_sql('ClaimsMooshSet', con, schema='Temp', index=False, chunksize=5000,  if_exists='replace')  
    
    rows1 = con.execute(text('select count(*) from Empire.ClaimsFlat_Test ')).fetchone()
    rows_count1 = rows1[0]


    rows2 = con.execute(text('select count(*) from Temp.ClaimsMooshSet')).fetchone()
    rows_count2 = rows2[0]

    print(rows_count1-rows_count2, " new rows")        
    
    #####Timing END
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    ##############################################################
    ###Create space
    
    cursor.execute('DROP TABLE IF EXISTS Temp.HfClaimCheck;')
    cursor.execute('DROP TABLE IF EXISTS Temp.ClaimsMooshA;')
    cursor.execute('DROP TABLE IF EXISTS Temp.ClaimsMooshFlag;')
    cursor.execute('DROP TABLE IF EXISTS Temp.ClaimsMooshCte;')
    
    ##############################################################
    
    #Run Stored Procedure
    logger.info('{0} Running Moosh SP'.format(time.ctime()))    
    itemISthere = conn.cursor()
    itemISthere.execute("Empire.SpMoosh_load")
    logger.info('{0} Moosh SP run successful'.format(time.ctime()))      
    
    cursor.execute('DROP TABLE IF EXISTS Temp.ClaimsMooshSet;')
    
    #####Timing END
    end_time = time.monotonic()
    print(timedelta(seconds=end_time - start_time))
    
    print('pymoosh complete Dun Dun DUNNN!')
    
    #conn.close()
    #con.dispose()
    
    logger.info('{0} Moosh complete!'.format(time.ctime()))  
    
    cursor.close()
    
#####Timing 
current_time = datetime.datetime.now().strftime("%H:%M:%S")
print("Current time:", current_time)
