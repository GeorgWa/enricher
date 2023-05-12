import gradio as gr
import random
import time
import numpy as np
from unipressed import IdMappingClient, UniprotkbClient
import time
import pandas as pd
import requests
from io import StringIO
import openai

intro_prompt = "You are an AI assistant helpful for clinical research. You will be provided a list of proteins which were identified as upregulated in a proteomics experiment. \n"
final_prompt = """ Use the provided information to give a hypothesis for the observed pattern. 
Only use scientifically validated facts and mention the relevant protein names. Always mention the gene name in parentheses after the protein. Don't cite literature in the response. Don't give a summary of the enrichment analysis if it's not needed for the hypothesis. Structure the response in multiple paragraphs and use markdown if necessary. \n"""

def get_enrichment(uniprot_list):

    fg = "%0d".join(uniprot_list)
    result = requests.post(r"https://agotool.org/api_orig",
                    params={"output_format": "tsv",
                            "enrichment_method": "genome",
                            "limit_2_entity_type": "-20;-21;-22;-23;-25;-26;-52;-57;-58",
                            "taxid": 9606},
                    data={"foreground": fg})
    df = pd.read_csv(StringIO(result.text), sep='\t')
    return df.sort_values("effect_size", ascending=False)

def build_enrichment_prompt(enrichment_df):

    str = "For the given set of proteins a gene set enrichment analysis was performed. The following results were obtained:\n"

    for (etype, label) in [
            ([-20], 'Gene Ontology: Cellular Component'),
            ([-21], 'Gene Ontology: Biological Process'),
            ([-23], 'Gene Ontology: Molecular Function'),
            ([-25], 'Brenda Tissue Ontology'),
            ([-26], 'Disease Ontology'),
            ([-57], 'Reactome Pathway'),
        ]:

        current_enrichment_df = enrichment_df[enrichment_df['etype'].isin(etype)]

        if current_enrichment_df.shape[0] > 0:
            str += f"\n{label}: \n"
            
            for i, row in current_enrichment_df.head(5).iterrows():
                enrichment =(row['foreground_count'] / row['foreground_n'])/ (row['background_count'] / row['background_n'])
                str += f"{row['description']} ({enrichment:.1f} fold enriched)\n"
        
    return str

def build_protein_prompt(protein_df):

    str = "\n Proteins are given with the highest fold change first:\n"

    for i, row in protein_df.head(20).iterrows():
        str += f"{row['full_name']} {row['gene_name']} ({row['accession']})\n"

    return str

def map_uniprot(protein_df):

    request = IdMappingClient.submit(
        source="GeneCards", dest="UniProtKB", ids=set(protein_df['raw_name'].values)
    )

    for i in range(100):
        status = request.get_status()
        
        time.sleep(0.5)

        if status == 'FINISHED':
            mapped_list = list(request.each_result())
            break

    result_df = pd.DataFrame(mapped_list)
    result_df = result_df.rename(columns={'from':'raw_name','to':'uniprot_id'})
    
    # fill in missing values with None
    protein_df = protein_df.merge(result_df, on='raw_name', how='left')
    protein_df['label'] = 'Success'
    protein_df.loc[protein_df['uniprot_id'].isnull(), 'label'] = 'Failed'

    return protein_df

def parse_proteins(protein_csv):

    sep = ''
    for c_sep in [';',',','\t',' ']:
        if c_sep in protein_csv:
            sep = c_sep
            break

    protein_names = protein_csv.split(sep)
    protein_names = [x.strip() for x in protein_names]
    protein_names = [x.upper() for x in protein_names]

    protein_df = pd.DataFrame({'raw_name': protein_names}) 
    return map_uniprot(protein_df)

def fetch_annotation(protein_df):

    request = UniprotkbClient.fetch_many(
        protein_df['uniprot_id'].values
    )

    out_df = []

    for row, response in zip(protein_df.to_dict(orient="records"), request):

        print(response)

        try:
            row['full_name'] = response['proteinDescription']['recommendedName']['fullName']['value']
        except:
            row['full_name'] = ''

        try:
            row['gene_name'] = response['genes'][0]['geneName']['value']
        except:
            row['gene_name'] = ''

        try:
            row['accession'] = response['primaryAccession']

        except:
            row['accession'] = ''

        out_df.append(row)

    out_df = pd.DataFrame(out_df)
    return out_df

def perform_analysis(text_input, protein_input):

    protein_df = parse_proteins(protein_input)

    protein_validated_df = protein_df[protein_df['label']=='Success']
    protein_annotated_df = fetch_annotation(protein_validated_df)

    protein_prompt = build_protein_prompt(protein_annotated_df)
    enrichment_df = get_enrichment(list(protein_annotated_df['uniprot_id'].values))
    enrichment_prompt = build_enrichment_prompt(enrichment_df)

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": intro_prompt},
            {"role": "user", "content": text_input},
            {"role": "system", "content": protein_prompt},
            {"role": "system", "content": enrichment_prompt},
            {"role": "user", "content": final_prompt}
            
        ]
    )

    print(response)

    highlighted_text = []
    for row in protein_df.to_dict(orient="records"):
        highlighted_text.append((row['raw_name'], row['label']))

    return highlighted_text, response['choices'][0]['message']['content']


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    text_input = gr.Textbox(label='Experimental setup', lines=3, placeholder="Describe your experimental setup and under which conditions you have observed the proteins")
    protein_input = gr.Textbox(label='Differntially expressed proteins', lines=1, placeholder="Enter protein names separated by comma, semicolon, tab or space")
    protein_validation = gr.HighlightedText().style(color_map={"Failed": "red", "Success": "green"})
    text_output = gr.Textbox(label='Results', lines=10, placeholder="Results will be shown here")
    text_button = gr.Button("Run")

    text_button.click(perform_analysis, inputs=[text_input,protein_input], outputs=[protein_validation,text_output])

    gr.Examples(
        [ 
            [
                "I investigated factors driving metastasis in low grade serous ovarian cancer (LGSOC). The following proteins were upregulated in metastatic lesions compared to primary tumors:" ,
                "COL6A2 COL6A3 TPD52L2 PTGFRN ITGAV NENF FGA POSTN GOLM2 COL6A1 LUC7L2 ACADM SCAMP2"
                ],
        ],
        [text_input,protein_input], 
        protein_validation,
        perform_analysis
    )

    
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_port=8000)