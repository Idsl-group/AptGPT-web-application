import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode

cell_button_renderer = JsCode("""
class BtnCellRenderer {
    init(params) {
        this.params = params;
        this.eGui = document.createElement('button');
        this.eGui.innerText = 'View';
        this.eGui.classList.add('btn', 'btn-primary');
        this.eGui.style.width = '70px';
        this.eGui.style.height = '30px';
        this.eGui.addEventListener('click', () => {
            params.api.deselectAll();
            params.api.selectNode(params.node, true, true);
        });
    }
    getGui() {
        return this.eGui;
    }
}
""")

def view_table(df):
    df['View Structure'] = ''
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_selection('single', use_checkbox=False)

    for col in df.columns:
        if col == 'View Structure':
            gb.configure_column(col, header_name='', suppressSizeToFit=True, cellRenderer='BtnCellRenderer')

        gb.configure_column(col, type='rightAligned', editable=False)

    grid_options = gb.build()
    grid_options['components'] = {
        'BtnCellRenderer': cell_button_renderer.js_code
    }
    grid_options['rowHeight'] = 40

    grid_response = AgGrid(df, gridOptions=grid_options, update_mode=GridUpdateMode.MODEL_CHANGED, width='100%', fit_columns_on_grid_load=True, allow_unsafe_jscode=True)

    if grid_response['selected_rows']:
        selected = grid_response['selected_rows'][0]
        st.toast('Selected')
        with st.modal("Secondary Structure"):
            st.write(selected)