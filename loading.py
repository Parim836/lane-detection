import gradio as gr

def render_loading(percent=None, text="Loading..."):

    percent_text = "" if percent is None else f"{percent}%"

    return f"""
    </style>
    <div style="
        position:fixed;
        top:0;
        left:0;
        width:100%;
        height:100%;
        background:rgba(0,0,0,0.5);
        z-index:9999;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    ">

        <div style="
            width:80px;
            height:80px;
            border:10px solid #f3f3f3;
            border-top:10px solid #3498db;
            border-radius:50%;
            animation: spin 1s linear infinite;
        "></div>

        <p style="color:white; margin-top:20px; font-size:20px;">
            {text} 
        </p>

    </div>
    """

        