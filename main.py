import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from Components.chat_logic import get_streaming_response
from Components.qdrant_store import store_question_response
from contextlib import contextmanager
from nicegui import ui, app

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

@ui.page('/')
def main():
    ui.add_head_html('''
    <style>
        .custom-red-background {
            background-color: rgb(194, 0, 1) !important;
        }

        .custom-red-color {
            color: rgb(194, 0, 1) !important;
        }
        .sent-message .q-message-text {
            background-color: rgb(194, 0, 1);
        }
        .sent-message .q-message-text:last-child:before {
            border-bottom-color: rgb(194, 0, 1);
        }
        .sent-message .q-message-text-content {
            color: white;
        }

        .received-message .q-message-text {
            background-color: white;
            border-style: solid;
            border-color: red;
        }
        .received-message .q-message-text:last-child:before {
            border-bottom-color: white;
        }
        .no-hover-bg.q-btn:hover {
            background-color: transparent !important;
        }
        .dialog-container {
            max-height: 400px;
            overflow-y: auto;
            overflow-x: hidden;
            width: 100%;
        }
        
    </style>
    ''')

    async def send(text, message_container, send_button, dialog) -> None:
        question = text.value
        if not question:
            return

        timestamp = datetime.now().isoformat()
        text.value = ''
        with message_container:
            ui.chat_message(text=question, sent=True).classes('sent-message')
            response_message = ui.chat_message(sent=False).classes('received-message')
            spinner = ui.spinner(type='dots')

        @contextmanager
        def disable_with_spinner(button: ui.button):
            original_icon = button._props.get('icon', '')
            button.set_icon('autorenew')
            button.disable()
            try:
                yield
            finally:
                button.set_icon(original_icon)
                button.enable()

        with disable_with_spinner(send_button):
            try:
                response = ""
                response_message.clear()
                with response_message:
                    response_element = ui.html("")
                
                async for chunk in get_streaming_response(question):
                    if chunk.startswith("Error:"):
                        response = chunk
                        break
                    response += chunk
                    
                    response_element.content = response.replace('\n', '<br>')
                    ui.run_javascript('''
                        let container = document.querySelector(".dialog-container");
                        container.scrollTop = container.scrollHeight;
                    ''')

                message_container.remove(spinner)

            except Exception as e:
                response_message.clear()
                with response_message:
                    ui.html(f"Error: Failed to process query - {str(e)}")
            
            finally:
                if spinner in message_container:
                    message_container.remove(spinner)

    # Styling
    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')
    ui.query('.q-page').classes('flex items-center justify-center')
    ui.query('.nicegui-content').classes('w-full')

    # Suppress default "Connection lost" message
    ui.run_javascript("""
        window.addEventListener('beforeunload', () => {});
        window.nicegui.handleDisconnect = () => {};
    """)

    # Main page content
    with ui.column().classes('w-full max-w-3xl mx-auto my-6 items-center'):
        ui.button("Let's Chat", icon='forum', on_click=lambda: dialog.open()).classes('custom-red-background text-white h-11 rounded-lg normal-case absolute bottom-10 right-10')
        # ui.button('Open Chat', on_click=lambda: dialog.open()).classes('custom-red-background text-white')

    # Dialogue box
    with ui.dialog() as dialog, ui.card().classes('w-full max-w-lg'):
        ui.label('E&E Solutions').classes('text-h6 custom-red-color')
        # Close button
        ui.label('X').classes('cursor-pointer custom-red-color absolute top-5 right-5').on(
            'click', lambda: dialog.close()
        )
        message_container = ui.element('div').classes('dialog-container')
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = 'Ask anything...'
            with ui.input(placeholder=placeholder) \
                    .props('rounded outlined input-class=mx-3 color=red') \
                    .classes('w-full self-center') as text:


                # Send on Enter key press
                text.on('keydown.enter', lambda e: send(text, message_container, send_button, dialog))

                # Send on button click
                send_button = ui.button(icon='send', on_click=lambda: send(text, message_container, send_button, dialog)) \
                    .props('flat dense').classes('custom-red-color').props('slot=append')

ui.run(
    title='E&E Solutions',
)
