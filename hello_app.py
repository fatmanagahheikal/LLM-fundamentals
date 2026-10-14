import gradio as gr

def greet(name):
    return f"Hello, {name}! Welcome to Gradio."

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text",
    title="My First Gradio App 👋",
)

demo.launch()