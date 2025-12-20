# import gradio as gr
# from dr_app.src.inference import run_all_models


# def predict(img):
#     df, cam1, cam2, cam3 = run_all_models(img)
#     return df, cam1, cam2, cam3

# demo = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil", label="Upload retinal fundus image"),
#     outputs=[
#         gr.Dataframe(label="Predictions"),
#         gr.Image(label="ResNet-50 Grad-CAM"),
#         gr.Image(label="EfficientNet-B2 Grad-CAM"),
#         gr.Image(label="ViT-B/16 Explanation"),
#     ],
#     title="DR Grading Demo",
# )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)
