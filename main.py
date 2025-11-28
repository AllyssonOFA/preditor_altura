import gradio as gr
import pickle

from Logic.predicao_altura import prever_valor, criar_modelo
from Logic.definitions import MODEL_FILE


try:
    modelo, scaler, metricas = pickle.load(open(MODEL_FILE, 'rb'))
except:
    criar_modelo()
    modelo, scaler, metricas = pickle.load(open(MODEL_FILE, 'rb'))

def prever(valor):
    return prever_valor(modelo, scaler, valor)


def main():
    with gr.Blocks(title='Preditor de Altura') as app:
        gr.Markdown(
            """
            # Previsor de Altura
            Preveja a altura de um filho com base na altura de seu pai.  
            """
        )
        inp = gr.Number(label='Altura do Pai (metros)', placeholder="Altura do pai, em metros:")
        out = gr.Text(label='Altura do Filho (metros)', placeholder='Altura do filho, em metros:')
        
        with gr.Accordion("Métricas do modelo:", open=False):
            gr.Markdown(f"""
                RMSE: {metricas['RMSE']:.4f} \n
                MAE: {metricas['MAE']:.4f} \n
                MAPE: {metricas['MAPE']:.4f}% \n
                R²: {metricas['R2']:.4f}
            """)

        btn = gr.Button("Prever")
        btn.click(fn = prever, inputs=inp, outputs=out)
        gr.Markdown(container=True)
    app.launch(favicon_path='Media/icons-ruler.png', share=True)

if __name__ == '__main__':
    main()
