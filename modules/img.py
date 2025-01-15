from shiny import App, reactive, render, ui, run_app, Session, Outputs, module
from shiny.types import ImgData

@module.ui
def img_ui():
    return ui.card(
        ui.input_checkbox('checkbox', ''),
        ui.output_image('image'),
        fill=False
    )

@module.server
def img_server(input, output, session, file, selected_images):

    @render.image
    def image():
        img: ImgData = {"src": str(file), "width":"100%"}
        print(file)
        return img


    @reactive.Effect
    @reactive.event(input.checkbox)
    def selected():
        # Get the current state of selected_images
        current_selected_images = selected_images.get()

        # Update the selection status of this image
        if input.checkbox():
            current_selected_images[file] = True
        else:
            if file in current_selected_images:
                del current_selected_images[file]

        # Set the updated state back to selected_images
        selected_images.set(current_selected_images)