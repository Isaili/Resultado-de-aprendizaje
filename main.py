import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Análisis Exploratorio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataAnalyzer:
    """Clase principal para el análisis de datos"""
    
    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.column_info = {}
        
    def load_data(self, files):
        """Carga y concatena múltiples archivos CSV"""
        dataframes = []
        
        for file in files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
                st.success(f"✅ Archivo {file.name} cargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error cargando {file.name}: {str(e)}")
                
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            self.filtered_data = self.data.copy()
            self._profile_columns()
            return True
        return False
    
    def _profile_columns(self):
        """Perfila automáticamente las columnas del dataset"""
        for col in self.data.columns:
            col_data = self.data[col]
            
            # Detectar tipo de columna
            if pd.api.types.is_numeric_dtype(col_data):
                col_type = "Numérico"
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                col_type = "Fecha"
            elif col_data.dtype == 'bool':
                col_type = "Booleano"
            else:
                col_type = "Categórico"
                
            # Intentar convertir a datetime si parece una fecha
            if col_type == "Categórico":
                try:
                    pd.to_datetime(col_data.dropna().head(100))
                    col_type = "Fecha (Texto)"
                except:
                    pass
            
            self.column_info[col] = {
                'tipo': col_type,
                'nulos': col_data.isnull().sum(),
                'unicos': col_data.nunique(),
                'dtype': str(col_data.dtype)
            }
    
    def get_column_info_df(self):
        """Retorna información de columnas como DataFrame"""
        info_data = []
        for col, info in self.column_info.items():
            info_data.append({
                'Columna': col,
                'Tipo': info['tipo'],
                'Valores Nulos': info['nulos'],
                'Valores Únicos': info['unicos'],
                'Tipo de Dato': info['dtype']
            })
        return pd.DataFrame(info_data)
    
    def apply_filters(self, filters):
        """Aplica filtros al dataset"""
        self.filtered_data = self.data.copy()
        
        for filter_config in filters:
            col = filter_config['column']
            filter_type = filter_config['type']
            values = filter_config['values']
            
            if filter_type == 'numeric_range':
                self.filtered_data = self.filtered_data[
                    (self.filtered_data[col] >= values[0]) & 
                    (self.filtered_data[col] <= values[1])
                ]
            elif filter_type == 'categorical':
                self.filtered_data = self.filtered_data[
                    self.filtered_data[col].isin(values)
                ]
            elif filter_type == 'date_range':
                self.filtered_data = self.filtered_data[
                    (pd.to_datetime(self.filtered_data[col]) >= values[0]) &
                    (pd.to_datetime(self.filtered_data[col]) <= values[1])
                ]
    
    def sample_data(self, method, **kwargs):
        """Aplica métodos de partición de datos"""
        if method == "random":
            fraction = kwargs.get('fraction', 0.1)
            self.filtered_data = self.filtered_data.sample(frac=fraction, random_state=42)
        elif method == "stratified":
            col = kwargs.get('column')
            fraction = kwargs.get('fraction', 0.1)
            if col in self.filtered_data.columns:
                self.filtered_data = self.filtered_data.groupby(col).apply(
                    lambda x: x.sample(frac=fraction, random_state=42)
                ).reset_index(drop=True)
        elif method == "temporal":
            col = kwargs.get('column')
            days = kwargs.get('days', 30)
            if col in self.filtered_data.columns:
                date_col = pd.to_datetime(self.filtered_data[col])
                cutoff_date = date_col.max() - timedelta(days=days)
                self.filtered_data = self.filtered_data[date_col >= cutoff_date]

def create_basic_plot(data, plot_type, x_col=None, y_col=None, hue_col=None, size_col=None):
    """Crea gráficos básicos según el tipo especificado"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if plot_type == "histogram":
        if x_col:
            ax.hist(data[x_col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(x_col)
            ax.set_ylabel('Frecuencia')
            ax.set_title(f'Histograma de {x_col}')
            
    elif plot_type == "boxplot":
        if y_col and x_col:
            sns.boxplot(data=data, x=x_col, y=y_col, ax=ax)
        elif y_col:
            sns.boxplot(data=data, y=y_col, ax=ax)
        ax.set_title(f'Boxplot de {y_col}')
        
    elif plot_type == "violin":
        if y_col and x_col:
            sns.violinplot(data=data, x=x_col, y=y_col, ax=ax)
        elif y_col:
            sns.violinplot(data=data, y=y_col, ax=ax)
        ax.set_title(f'Violin Plot de {y_col}')
        
    elif plot_type == "scatter":
        if x_col and y_col:
            if hue_col:
                sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                ax.scatter(data[x_col], data[y_col], alpha=0.6)
            ax.set_title(f'Scatter Plot: {x_col} vs {y_col}')
            
    elif plot_type == "bar":
        if x_col:
            if y_col:
                data.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
                ax.set_title(f'Promedio de {y_col} por {x_col}')
            else:
                data[x_col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Frecuencia de {x_col}')
        
    elif plot_type == "heatmap":
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matriz de Correlación')
            
    elif plot_type == "pie":
        if x_col:
            value_counts = data[x_col].value_counts()
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.set_title(f'Gráfico Circular de {x_col}')
    
    plt.tight_layout()
    return fig

def create_advanced_plot(data, plot_type, columns=None):
    """Crea gráficos avanzados"""
    fig = plt.figure(figsize=(12, 8))
    
    if plot_type == "pairplot":
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]  # Limitar a 5 para rendimiento
        if len(numeric_cols) > 1:
            g = sns.pairplot(data[numeric_cols])
            return g.fig
            
    elif plot_type == "diverging_bars":
        if columns and len(columns) >= 2:
            x_col, y_col = columns[0], columns[1]
            grouped_data = data.groupby(x_col)[y_col].mean().sort_values()
            
            colors = ['red' if x < 0 else 'blue' for x in grouped_data.values]
            ax = fig.add_subplot(111)
            ax.barh(range(len(grouped_data)), grouped_data.values, color=colors)
            ax.set_yticks(range(len(grouped_data)))
            ax.set_yticklabels(grouped_data.index)
            ax.set_xlabel(y_col)
            ax.set_title(f'Barras Divergentes: {y_col} por {x_col}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
    elif plot_type == "slope_chart":
        # Implementación simplificada de slope chart
        if columns and len(columns) >= 3:
            cat_col, val_col, time_col = columns[0], columns[1], columns[2]
            pivot_data = data.pivot_table(values=val_col, index=cat_col, columns=time_col, aggfunc='mean')
            
            if pivot_data.shape[1] >= 2:
                ax = fig.add_subplot(111)
                for idx, row in pivot_data.iterrows():
                    values = row.dropna()
                    if len(values) >= 2:
                        ax.plot([0, 1], [values.iloc[0], values.iloc[1]], 'o-', alpha=0.7, label=idx)
                
                ax.set_xlim(-0.1, 1.1)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([str(pivot_data.columns[0]), str(pivot_data.columns[1])])
                ax.set_ylabel(val_col)
                ax.set_title(f'Slope Chart: {val_col} por {cat_col}')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def save_plot_as_image(fig, filename="plot", format="png", dpi=300):
    """Guarda el gráfico como imagen y retorna el enlace de descarga"""
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    
    if format == "png":
        img_str = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{img_str}" download="{filename}.png">📥 Descargar PNG</a>'
    else:  # PDF
        img_str = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:application/pdf;base64,{img_str}" download="{filename}.pdf">📥 Descargar PDF</a>'
    
    return href

def main():
    st.title("📊 Dashboard Interactivo de Análisis Exploratorio")
    st.markdown("---")
    
    # Inicializar el analizador
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DataAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar para controles
    st.sidebar.header("🔧 Controles")
    
    # Carga de datos
    st.sidebar.subheader("📂 Carga de Datos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecciona archivos CSV",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.sidebar.button("Cargar Datos"):
            if analyzer.load_data(uploaded_files):
                st.sidebar.success(f"✅ {len(uploaded_files)} archivos cargados")
                st.sidebar.info(f"Total de registros: {len(analyzer.data)}")
                st.sidebar.info(f"Total de columnas: {len(analyzer.data.columns)}")
    
    # Verificar si hay datos cargados
    if analyzer.data is not None:
        
        # Mostrar información de columnas
        st.header("📋 Información de Columnas")
        col_info_df = analyzer.get_column_info_df()
        st.dataframe(col_info_df, use_container_width=True)
        
        # Selección de columnas para incluir
        st.sidebar.subheader("📊 Selección de Columnas")
        selected_columns = st.sidebar.multiselect(
            "Columnas a incluir en el análisis:",
            options=list(analyzer.data.columns),
            default=list(analyzer.data.columns)
        )
        
        # Filtros dinámicos
        st.sidebar.subheader("🔍 Filtros Dinámicos")
        filters = []
        
        # Filtros para columnas numéricas
        numeric_cols = [col for col, info in analyzer.column_info.items() 
                       if info['tipo'] == 'Numérico' and col in selected_columns]
        
        for col in numeric_cols[:3]:  # Limitar a 3 filtros
            min_val = float(analyzer.data[col].min())
            max_val = float(analyzer.data[col].max())
            
            range_vals = st.sidebar.slider(
                f"Rango para {col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                key=f"range_{col}"
            )
            
            if range_vals != (min_val, max_val):
                filters.append({
                    'column': col,
                    'type': 'numeric_range',
                    'values': range_vals
                })
        
        # Filtros para columnas categóricas
        categorical_cols = [col for col, info in analyzer.column_info.items() 
                           if info['tipo'] == 'Categórico' and col in selected_columns]
        
        for col in categorical_cols[:2]:  # Limitar a 2 filtros
            unique_vals = analyzer.data[col].unique()
            if len(unique_vals) <= 20:  # Solo mostrar si hay pocas categorías
                selected_vals = st.sidebar.multiselect(
                    f"Valores para {col}:",
                    options=unique_vals,
                    default=unique_vals,
                    key=f"cat_{col}"
                )
                
                if len(selected_vals) != len(unique_vals):
                    filters.append({
                        'column': col,
                        'type': 'categorical',
                        'values': selected_vals
                    })
        
        # Aplicar filtros
        if filters:
            analyzer.apply_filters(filters)
            st.sidebar.info(f"Registros después de filtros: {len(analyzer.filtered_data)}")
        
        # Métodos de partición
        st.sidebar.subheader("🎯 Partición de Datos")
        partition_method = st.sidebar.selectbox(
            "Método de partición:",
            ["Ninguno", "Muestreo Aleatorio", "Muestreo Estratificado", "Partición Temporal"]
        )
        
        if partition_method == "Muestreo Aleatorio":
            fraction = st.sidebar.slider("Fracción de muestra:", 0.01, 1.0, 0.1)
            analyzer.sample_data("random", fraction=fraction)
            
        elif partition_method == "Muestreo Estratificado":
            strat_col = st.sidebar.selectbox("Columna para estratificar:", categorical_cols)
            fraction = st.sidebar.slider("Fracción de muestra:", 0.01, 1.0, 0.1)
            if strat_col:
                analyzer.sample_data("stratified", column=strat_col, fraction=fraction)
                
        elif partition_method == "Partición Temporal":
            date_cols = [col for col, info in analyzer.column_info.items() 
                        if 'Fecha' in info['tipo'] and col in selected_columns]
            if date_cols:
                date_col = st.sidebar.selectbox("Columna de fecha:", date_cols)
                days = st.sidebar.number_input("Últimos N días:", min_value=1, value=30)
                analyzer.sample_data("temporal", column=date_col, days=days)
        
        # Mostrar estadísticas del dataset activo
        st.header("📈 Estadísticas del Dataset Activo")
        st.info(f"Registros activos: {len(analyzer.filtered_data)} | Columnas: {len(selected_columns)}")
        
        # Resumen estadístico
        numeric_data = analyzer.filtered_data[selected_columns].select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.subheader("📊 Resumen Estadístico")
            summary = numeric_data.describe()
            st.dataframe(summary, use_container_width=True)
            
            # Botón para descargar resumen
            csv_summary = summary.to_csv()
            st.download_button(
                label="📥 Descargar Resumen (CSV)",
                data=csv_summary,
                file_name="resumen_estadistico.csv",
                mime="text/csv"
            )
        
        # Generación de gráficos
        st.header("📊 Generación de Gráficos")
        
        # Crear tabs para diferentes tipos de gráficos
        tab1, tab2, tab3 = st.tabs(["Gráficos Básicos", "Gráficos Acoplados", "Gráficos Avanzados"])
        
        with tab1:
            st.subheader("Gráficos Básicos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                plot_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["histogram", "boxplot", "violin", "scatter", "bar", "heatmap", "pie"]
                )
                
                x_col = st.selectbox("Columna X:", [None] + selected_columns)
                y_col = st.selectbox("Columna Y:", [None] + selected_columns)
                hue_col = st.selectbox("Color por:", [None] + selected_columns)
            
            with col2:
                if st.button("Generar Gráfico Básico"):
                    try:
                        fig = create_basic_plot(
                            analyzer.filtered_data,
                            plot_type,
                            x_col,
                            y_col,
                            hue_col
                        )
                        st.pyplot(fig)
                        
                        # Opciones de descarga
                        col_a, col_b = st.columns(2)
                        with col_a:
                            png_link = save_plot_as_image(fig, f"{plot_type}_plot", "png")
                            st.markdown(png_link, unsafe_allow_html=True)
                        with col_b:
                            pdf_link = save_plot_as_image(fig, f"{plot_type}_plot", "pdf")
                            st.markdown(pdf_link, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error generando gráfico: {str(e)}")
        
        with tab2:
            st.subheader("Gráficos Acoplados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                plot1_type = st.selectbox("Primer gráfico:", ["histogram", "boxplot", "scatter"])
                plot2_type = st.selectbox("Segundo gráfico:", ["histogram", "boxplot", "scatter"])
                
            with col2:
                subplot_x = st.selectbox("Columna X (ambos):", selected_columns, key="subplot_x")
                subplot_y = st.selectbox("Columna Y (ambos):", selected_columns, key="subplot_y")
            
            if st.button("Generar Gráficos Acoplados"):
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Primer subplot
                    if plot1_type == "histogram":
                        ax1.hist(analyzer.filtered_data[subplot_x].dropna(), bins=30, alpha=0.7)
                        ax1.set_title(f'Histograma de {subplot_x}')
                    elif plot1_type == "boxplot":
                        analyzer.filtered_data.boxplot(column=subplot_y, ax=ax1)
                        ax1.set_title(f'Boxplot de {subplot_y}')
                    elif plot1_type == "scatter":
                        ax1.scatter(analyzer.filtered_data[subplot_x], analyzer.filtered_data[subplot_y], alpha=0.6)
                        ax1.set_title(f'Scatter: {subplot_x} vs {subplot_y}')
                    
                    # Segundo subplot
                    if plot2_type == "histogram":
                        ax2.hist(analyzer.filtered_data[subplot_y].dropna(), bins=30, alpha=0.7)
                        ax2.set_title(f'Histograma de {subplot_y}')
                    elif plot2_type == "boxplot":
                        analyzer.filtered_data.boxplot(column=subplot_x, ax=ax2)
                        ax2.set_title(f'Boxplot de {subplot_x}')
                    elif plot2_type == "scatter":
                        ax2.scatter(analyzer.filtered_data[subplot_y], analyzer.filtered_data[subplot_x], alpha=0.6)
                        ax2.set_title(f'Scatter: {subplot_y} vs {subplot_x}')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Opciones de descarga
                    col_a, col_b = st.columns(2)
                    with col_a:
                        png_link = save_plot_as_image(fig, "coupled_plots", "png")
                        st.markdown(png_link, unsafe_allow_html=True)
                    with col_b:
                        pdf_link = save_plot_as_image(fig, "coupled_plots", "pdf")
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error generando gráficos acoplados: {str(e)}")
        
        with tab3:
            st.subheader("Gráficos Avanzados")
            
            advanced_plot_type = st.selectbox(
                "Tipo de gráfico avanzado:",
                ["pairplot", "diverging_bars", "slope_chart"]
            )
            
            if advanced_plot_type == "pairplot":
                if st.button("Generar Pairplot"):
                    try:
                        fig = create_advanced_plot(analyzer.filtered_data, "pairplot")
                        st.pyplot(fig)
                        
                        png_link = save_plot_as_image(fig, "pairplot", "png")
                        st.markdown(png_link, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generando pairplot: {str(e)}")
                        
            elif advanced_plot_type == "diverging_bars":
                col1, col2 = st.columns(2)
                with col1:
                    div_x = st.selectbox("Columna categórica:", categorical_cols)
                with col2:
                    div_y = st.selectbox("Columna numérica:", numeric_cols)
                
                if st.button("Generar Barras Divergentes"):
                    try:
                        fig = create_advanced_plot(
                            analyzer.filtered_data, 
                            "diverging_bars", 
                            [div_x, div_y]
                        )
                        st.pyplot(fig)
                        
                        png_link = save_plot_as_image(fig, "diverging_bars", "png")
                        st.markdown(png_link, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generando barras divergentes: {str(e)}")
        
        # Exportación de datos
        st.header("💾 Exportación de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Descargar Dataset Filtrado"):
                csv_data = analyzer.filtered_data.to_csv(index=False)
                st.download_button(
                    label="Descargar CSV",
                    data=csv_data,
                    file_name="dataset_filtrado.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.info(f"Filas en dataset filtrado: {len(analyzer.filtered_data)}")
    
    else:
        st.info("👆 Por favor, carga uno o más archivos CSV para comenzar el análisis.")
        
        # Mostrar datos de ejemplo
        st.header("📋 Ejemplo de Datos Requeridos")
        st.markdown("""
        **Requisitos del dataset:**
        - ≥ 10,000 registros
        - ≥ 20 variables/columnas
        - Formatos soportados: CSV
        
        **Tipos de columnas detectadas automáticamente:**
        - Numéricas (int, float)
        - Categóricas (string, object)
        - Fechas (datetime)
        - Booleanas (bool)
        """)

if __name__ == "__main__":
    main()