from langchain.document_loaders import PyPDFLoader
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


descr_event = """2. DESCRIPTION OF THE EVENT
The details upon the Out of Limit (OOL) event initiated for T22 Tank due to routine Environmental 
Monitoring (EM) are as indicated in Table 1. 
PR# Description Sample date Discovery Date Results 
3016700 OOL TOC 12 September 2022 15 September 2022 505 ppb
Table 1 Event description
The Total Organic Carbon (TOC) testing was performed according to SOP-015413 and results did 
not meet the acceptance criteria as defined in SOP-048762 (NMT 500 ppb). 
All others testing performed on EM samples, i.e. pH, conductivity, LAL and bioburden were within 
the specified limits (see Table 2).
DIW 
pre-rinsing
Alkaline 
Cleaning
WFI 
rinsing

Test Unit Result Acceptance criteria
pH - 5.6 5.0 – 7.0
Conductivity µS/cm 1.1 See SOP-026359
TOC ppb 505 ≤ 500
LAL EU/ml 0.005 ≤ 0.25
Bioburden Cfu/10 ml 0 < 1 
Table 2 Results of testing performed on TANK_T22 samples taken on 12 September 2022
(ref.: NWA database). 
The previous EM testing of TANK_T22 was performed on 11 August 2022 and all the results 
obtained were satisfactory.
The EM re-testing for TOC was performed on 21 September 2022. The TOC (50 ppb) result obtained 
was satisfactory.
 
T22 is a buffer tank in Purification L2 Post Nano area (R605), and is used during the manufacturing 
of Cuvitru to prepare the dialysis buffer and the NaCl buffer used to clean the UF605.
No lots were impacted by this event, as concluded in the impact assessment (see Evaluation 
PR3016700 attached in TW8)."""

rc_full_str = """5. ROOT CAUSE INVESTIGATION
An investigation was conducted according to 6M methodology in order to identify the root cause. Each step concerning equipment cleaning, sampling and testing will be further investigated.
An OoL TOC on a tank could be caused by :
    - Presence of residue of carbon residue from the lot previously produced, due to a failed CIP
    - Contamination of the tank through the water used for CIP (WFI C102)
    - Contamination of the tank through the water used for pH-meter flush and sanitization (WFI C104)
    - Contamination of the tank following a maintenance activity
    - Contamination of the sample during sampling, the sampling point or the sampling equipment
    - Testing error
5.1. ENVIRONMENT
5.1.1. Analysis of CIP cycle
T22 is a non-protein, buffer tank in Purification L2 Post-Nano area and is used during the manufacturing of Cuvitru to prepare the dialysis buffer and the NaCl buffer used to clean the UF605.
T22 CIP cycle was performed and achieved on 12 September 2022 at 17h00 prior to EM sampling performed on 12 September 2022 around 17h21. The CIP was launched after an idle time following the production of Cuvitru lot BE13C095Z (end of production on the 09 September 2022), and before the production of Cuvitru lot BE13C097Z.
The CIP In-line conductivity (ref.: OSIPI database) results before (12 September 2022) and after EM sampling (14 September 2022) were satisfactory (1.40 and 1.31 µS/cm, respectively - ≤ 5.0 µS/cm (Ref.: SOP-013469)).
The CIP cycle of T22 followed the CIP 638-7 sequence as detailed in IT specification (Ref.: B2382_P_D_03_7.0_Design Specification - CIP Buffer & CIP UF & SOP-054471), as shown in Table 3.
The CIP lasted about three hours and 30 minutes. This is longer than what is usually observed when compared with other CIP cycles of T22 Tank in OSIPI database (up to 1h30 minutes). After the cleaning base step, the tank T22 is drained. When T22 weight is below 10kg, the final rinsing step is started. On 12 September 2022, after the cleaning base step, T22 weight could not get below 11kg. Therefore, the final rinsing step could not be started which caused the longer CIP time. However, the cleaning base and final rinsing steps duration were usual and were compliant. There is no reason to think that this longer CIP time would be the cause of an OoL TOC or a failed cleaning.
No alarms were reported during this CIP.
The condition of "NaOH Injection (Conductivity CE49561> 89,4 mS/cm during 40 s)" and circulation of cleaning base solution during 15 minutes in T22 tank as well as final rinsing step were held correctly as required by ITS sequence (see Figures 3 & 4).
The final CIP In-line conductivity measured for the CIP cycle of T22 tank performed on 12 September 2022 was 1.40 µS/cm which is below the limit validated to release a CIP cycle for routine manufacturing, i.e. ≤ 5.0 µS/cm (Ref.: SOP-013469). Additionally, the trend of T22 tank final CIP In-line conductivity (11 August 2022 – 21 September 2022), measured after the CIP of T22 did not show any negative tendency, (see Figure 5).
The injection of NaOH for station CIP638-7 is controlled via conductivity probe LECE49561 and final CIP In-line conductivity via conductivity probe LECE49549. These conductivity probes were verified by the metrology on, respectively, 10 September 2022 & 08 September 2022, and it was satisfactory (see Figures 8 & 9). The next due dates for these conductivity probes are, respectively, 10 March 2023 & 08 March 2023. No non-conformance events concerning these conductivity probes were reported.
The temperature of tank circulation C2 is controlled via temperature probe LETT49573. This conductivity probe was verified by metrology on 08 September 2022, and it was satisfactory (see Figure 10). The next due date for this probe is 08 March 2023. No non-conformance event concerning this probe was reported.
Tanks P13, P12 and Pipe TP653-TP605A in area R605 are cleaned in place as well by CIP638-7 station. These tanks are compliant for TOC for the year before the OoL TOC on T22 (from 12 September 2021 to 23 September 2022), see Figure.
CIP 638-7 station is connected to WFI loop 102. WFI loop C102 is also used for the EM sampling of tank T22. WFI loop C102 TOC was checked for the period of 01 August 2022 till 26 September 2022. NWA data demonstrated that TOC parameters are satisfactory (see Figure 13), i.e. not more than (NMT) 500 ppb according to SOP-026359.
Based on the considerations above, it can be concluded that the CIP cycle of T22 tank performed on 12 September 2022 was valid and the final conductivity of T22 measured in-line was 1.98 µS/cm (ref.: OSIPI database) which is below the limit validated to release a CIP cycle for routine manufacturing, i.e. ≤ 5.0 µS/cm (see Figure 5). A potential cross-contamination is therefore excluded.
WFI loop C104 is used for sanitization and flush of pH-meters used on tanks in the R605 area. WFI loop C104 TOC was checked for the period of 01 August 2022 till 26 September 2022. NWA data demonstrated that TOC parameters are satisfactory (see Figure 14).
5.1.2. Analysis of the sampling
Based on OSIPI data, EM samples were taken on 12 September 2022 at about 17h21 after the CIP cycle of T22 tank, completed on 12 September 2022 at 17h00. No activities were reported in T22 between end CIP and EM sampling (see Figure 8). It can be noticed that the sampling was performed within 48h (which is the validated clean hold time) after the end of the CIP cycle as per SOP-026466.
The EM sampling was performed according to SOP-026466 as following:
At Transfer panel (TP) TP605A, connect elbow « TP605A-4 » from « WFI » to «In T22 » - Fill up T22 with WFI (set-point = 45 Kg) - At Transfer panel TP605, connect elbow « 605-Pré-T22 » at « Out T22 » - Start the sampling function on HMI by selecting command "PRELEV" to open the tank bottom valve and take EM cleaning samples at transfer panel TP605C
In addition to EM sampling, the flexible TP605A-4 is only used to fill the tanks of the R605 area with WFI. The flexible “605-Pré-T22” is only used for EM sampling of T22 tank.
Cleaning status of small material single elbow “605-Pré-T22” is ensured by manual cleaning. Material was cleaned as per SOP- 048680 “LE09NE03031 - NETTOYAGE MANUEL DU MATERIEL”(cfr LE99FLNE352 – BE13C097Z).
It can be concluded that sampling is not the root cause of the out-of-limit TOC value measured on EM monitoring sample of T22 taken on 12 September 2022.
5.1.3. Analysis of the Testing
The laboratory investigation was performed using the 6M methodology. No potential root cause was identified for PR3016700 (ref.: PR2996517). All others testing performed on EM samples, i.e. pH, conductivity, LAL and bioburden were within the specified limits (see Table 2).
5.2. MATERIAL / MACHINE / EQUIPMENT
A review of the maintenance activities on the basis of JDE database was performed on T22, WFI loops C102 & C104, CIP station 638-7 and room 605 (from 11 August 2022 to 21 September 2022).
This review of maintenance activities assessed (ref.: Annexe 1 attached PR3016700 in TW8) that T22, WFI loops C102 & C104, CIP station 638-7 and room 605 were fully operational and that their integrity was ensured for the EM sampling. No potential root cause was identified.
It can be concluded that Material / Machine / Equipment is not the root cause of PR3016700.
5.3. RAW MATERIALS
NaOH 29% is the cleaning agent used for CIP of T22 tank. No carbon in that composition.
The TOC tube used to sample tank T22 on 12 September 2022 corresponds to lot number 21277- 4627. This lot number has been quarantined on 14 October 2022 as part of PR3060455 - Atypical trend of TOC occurrence for EM samples WFI, tanks, UF and columns from 02/Oct/22 to 07/Oct/22. This PR is still under investigation at the time.
It can be concluded that raw materials is not the root cause of PR3016700. Nonetheless, deviation PR3060455 has been opened on 14 October 2022 for an atypical trend in the occurrence OoL TOC observed for EM samples. This PR is still under investigation at the time.
5.4. MAN & METHOD
The laboratory investigation was performed using the 6M methodology. No potential root cause was identified for PR3016700 (ref.: PR2996517). Note that all others testing performed on EM cleaning samples, i.e. pH, conductivity, LAL and bioburden were within the specified limits (see Table 2). According to the laboratory investigation (ref.: PR2996517), the EM operator and QC analyst involved in the OoL treated here were in order of training : - EM sampling was performed as required in SOP-026466 “Prelevements des eaux de rincages des tanks”
The tests were performed as described in SOP-015413 “ Measurement of Total Organic Carbon with the use of TOC Sievers”.
An analysis of the above mentioned SOP’s revealed that they are explicit, as the methodology to be followed for the collection of rinse samples and their measurement.
It can be concluded that Man is not the root cause of PR3016700.
5.5. CHANGE & NON-CONFORMANCE EVENTS REVIEW
The change & non-conformance events (ref.: Trackwise 8) review was performed on the period from 11 August 2022 (last EM testing satisfactory) to 21 September 2022 (satisfactory retest of EM sampling). The research criteria applied were defined as following: “Zone 605”, “Tank T22”, “CIP station 638 7”, WFI loop C102 & C104”. No other non-conformance events were reported for T22 tank and CIP station 638-7 for the revised here period based on TW8 database. No change regarding research criteria was implemented for the revised here period based on TW8 database. The review of the maintenance activities performed on T22 between 11 August 2022 and 12 September 2022 was performed in the Facilities assessment of PR3016700.
5.6. CONCLUSION ON THE ROOT CAUSE INVESTIGATION
The root cause investigation demonstrated that:
The final conductivity measured in-line for the CIP cycle of T22 performed on 12 September 2022 was satisfactory;
NaOH 29% is the cleaning agent used for CIP of T22 tank. No carbon in that composition;  The TOC of WFI loops C102 and C104 was NMT 500 ppb;  T22 tank as well as the CIP station 638-7 were operational and the integrity of these equipments was ensured during the EM sampling;
The EM sampling was performed according to the standard operating procedures (within 48 hours after the CIP cycle) and no activities were reported between CIP and EM sampling;
The QC laboratory testing analysis was satisfactory and no testing error was identified;  No change impacting T22 and /or CIP station 638-7 were implemented in the period of 11 August 2022 (last satisfactory EM testing) till 21 September 2022 (re-test due to PR3016700);  The revision of maintenance activities (from 11 August 2022 to 21 September 2022) demonstrated that no maintenance Work Order had impact on this OOL TOC;
No other non-conformance events concerning T22 were reported;  LAL, pH, conductivity and Bioburden results obtained for TANK_T22 taken on 12 September 2022 were satisfactory;
No assignable root cause was identified in the investigation of PR3016700.
Nonetheless, deviation PR3060455 has been opened on 14 October 2022 for an atypical trend in the occurrence OoL TOC observed for EM samples. This PR is still under investigation at the time."""


TEMPLATE_INSTR_CHAT_STR = (
    """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not reference any given instructions or context. Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<Context>
{context_str}

<Chat history for additional context>
{previous_str}

{query_str} [/INST]"""
)

TEMPLATE_INSTR_STR = (
    """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not reference any given instructions or context. Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<Context>
{context_str}

{query_str} [/INST]"""
)

TEMPLATE_SUMMARY_STR = (
    """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not speculate or make up information. Do not reference any given instructions or context. Use the following pieces of conversation to write a summary according to the instructions provided at the end.

{chat_str}
Based on this, write a comprehensive and well-structured summary written in a professional tone suitable for an academic audience. The summary should cover all the key points and main ideas presented in the original text, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. The length of the summary should be appropriate for the length and complexity of the original text, providing a clear and accurate overview without omitting any important information. [/INST]"""
)

TEMPLATE_VANILLA_MISTRAL = (
    """<s>[INST] You are a helpful, respectful and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not reference any given instructions or context. Use the following pieces of context to answer the query at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<Context>
{context_str}

{query_str} [/INST]"""
)

# If any of the information is not available in the provided text, please indicate that it is not specified. 

def build_history(previous_hist, current_q, current_a):
    new_histo = f'Human: {current_q}\nAssistant:{current_a}'
    if previous_hist:
        new_histo = f'{previous_hist}\n{new_histo}'
    else:
        pass
    return new_histo

def create_chat_from_list(chat_list):
    chat_str = ''

    for q, a in chat_list:
        chat_str += f'Human: {q}\nAssistant:{a}\n'

    return chat_str

if __name__ == '__main__':

    use_chat = True
    use_vanilla_mistral = False

    llm = LlamaCpp(
        model_path="/Users/alexandresokolow/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.001,
        top_p=0.95,
        top_k=1,
        n_ctx=8000,
        max_tokens=2048,
        n_gpu_layers=1,
        n_batch=512,
        seed=27,
        f16_kv=True,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    # loader = PyPDFLoader("pdf/sartro/ER-TS-11-131.pdf")
    # documents = loader.load_and_split()
    # current_doc = documents[2].page_content

    # loader = PyPDFLoader("pdf/tank/PR3016700_tank_report.pdf")
    # documents = loader.load_and_split()
    # current_doc = documents[4].page_content + '\n\n' + documents[5].page_content

    #current_doc = rc_full_str

    current_doc = descr_event

    ###### QUERIES ########
    queries = [
        'Specify the violated governing procedure',
        #'What are the specific requirements of the violated governing procedure that were not met ?',
        'Which requirements of the violated governing procedure were not met ?',
        'Who discovered that the governing procedure was violated ?',
        'When was the violated governing procedure discovered ?',
        'How was the violated governing procedure discovered ?',
        'When did the deviation occur ?',
        'Who was involved in the deviation ?',
        'What are the potentially impacted elements (e.g : lots) due to the violation of the governing procedure ?'
        ]
    
    ###### QUERY RC #########

    queries_rc_analysis = [
    'Detail the methodology and tools used for the root cause investigation.',
    'List the potential causes that were evaluated.',
    'For each potential causes, what are the rationale that led to selecting or not each cause as the root cause ?',
    #'If a root cause was identified, is it a recurring root cause ?',
    #'If a root cause was identified is recurring, explain why it has repeated.'
]
    #queries = queries_rc_analysis

    ###### HISTORY CHAT #######
    history_chat = list()
    histo_str = ''

    ###### RELEVANT DOC #######
    #current_doc = documents[2].page_content
    
    if use_vanilla_mistral:
        ###### VANILLA MISTRAL #######
        add_prompt = "{answer}</s>[INST] {new_query} [/INST]"

        init_prompt = PromptTemplate(template=TEMPLATE_VANILLA_MISTRAL, input_variables=["context_str", "query_str"])
        llm_chain = LLMChain(prompt=init_prompt, llm=llm, verbose=True)
        q = queries.pop(0)
        answer_round = llm_chain.run({'context_str':current_doc, 'query_str':q})
        history_chat.append((q, answer_round))
        previous_prompt = init_prompt.format(context_str=current_doc, query_str=q)
        TEMPLATE_HIST_STR_ROLL = previous_prompt + add_prompt

        for q in queries:
            roll_prompt = PromptTemplate(template=TEMPLATE_HIST_STR_ROLL, input_variables=["answer", "new_query"])
            llm_chain = LLMChain(prompt=roll_prompt, llm=llm, verbose=True)
            answer_round = llm_chain.run({'answer': history_chat[-1][1], 'new_query': q})
            roll_prompt_str = roll_prompt.format(answer=history_chat[-1][1], new_query=q)
            TEMPLATE_HIST_STR_ROLL = roll_prompt_str + add_prompt
            history_chat.append((q, answer_round))

    else:
        if use_chat:
            prompt_instruct = PromptTemplate(template=TEMPLATE_INSTR_CHAT_STR, input_variables=["context_str", "previous_str", "query_str"])
        else:
            prompt_instruct = PromptTemplate(template=TEMPLATE_INSTR_STR, input_variables=["context_str", "query_str"])
        
        llm_chain_instruct = LLMChain(prompt=prompt_instruct, llm=llm, verbose=True)

        for q in queries:
            if use_chat:
                answer_round = llm_chain_instruct.run({'context_str':current_doc, 'previous_str': histo_str, 'query_str':q})
                histo_str = build_history(histo_str, q, answer_round)
            else:
                answer_round = llm_chain_instruct.run({'context_str':current_doc, 'query_str':q})

            history_chat.append((q, answer_round))

    summary_prompt = PromptTemplate(template=TEMPLATE_SUMMARY_STR, input_variables=["chat_str"])
    llm_chain_summary = LLMChain(prompt=summary_prompt, llm=llm, verbose=True)
    summary_dev = llm_chain_summary.run({'chat_str': create_chat_from_list(history_chat)})
    print(summary_dev)