# 📊 Dashboard Interactivo de Análisis Exploratorio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Un dashboard web interactivo para realizar análisis exploratorio de datos de forma dinámica con conjuntos de datos complejos (≥10,000 registros y ≥20 variables).

## 🎯 Características Principales

- 🔍 **Perfilado automático** de datos con detección de tipos
- 📈 **Visualizaciones dinámicas** (histogramas, boxplots, scatter plots, heatmaps)
- 🎛️ **Filtrado interactivo** por múltiples criterios
- 📊 **Métodos de partición** (aleatorio, estratificado, temporal)
- 💾 **Exportación** de gráficos y datos procesados
- ⚡ **Rendimiento optimizado** para grandes datasets

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/dashboard-analisis-exploratorio.git
cd dashboard-analisis-exploratorio

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar el dashboard
streamlit run main.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📋 Dependencias

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.15.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

## 🎨 Funcionalidades

### 📊 Análisis de Datos
- **Detección automática** de tipos de columnas (numérico, categórico, fecha, booleano)
- **Estadísticas descriptivas** por columna
- **Identificación de valores nulos** y únicos
- **Tabla informativa** con métricas clave

### 📈 Visualizaciones Disponibles
| Tipo de Gráfico | Descripción | Parámetros |
|------------------|-------------|------------|
| Histograma | Distribución de variables numéricas | Bins configurables |
| Boxplot | Distribución y outliers | Agrupación por categorías |
| Scatter Plot | Relación bivariada | Color por categorías |
| Heatmap | Matriz de correlaciones | Anotaciones opcionales |
| Barras | Frecuencias y agregaciones | Orientación configurable |

### 🔧 Métodos de Partición

1. **🎲 Muestreo Aleatorio**
   - Porcentaje configurable (10-100%)
   - Semilla fija para reproducibilidad

2. **📊 Muestreo Estratificado**
   - Mantiene proporciones por categoría
   - Selección de columna estratificante

3. **📅 Partición Temporal**
   - Filtrado por últimos N días
   - Detección automática de fechas

### 🎛️ Filtrado Dinámico
- ➡️ **Rangos numéricos** con sliders interactivos
- 🏷️ **Selección múltiple** para variables categóricas
- 📅 **Rangos de fechas** para variables temporales
- 🔄 **Aplicación automática** a todas las visualizaciones

## 💾 Exportación

### Gráficos
- Formato PNG con 300 DPI
- Descarga individual
- Nombres descriptivos automáticos

### Datos
- Subconjuntos filtrados en CSV
- Resúmenes estadísticos
- Preservación de filtros aplicados

## 🎯 Dominio de Aplicación

Optimizado para datos de **ventas y comercio electrónico**:
- 📅 Información temporal (fechas de ventas)
- 🏷️ Categorías de productos
- 🌍 Datos geográficos (regiones)
- 💰 Métricas financieras (ventas, ganancias, descuentos)
- 👥 Demografía de clientes (edad, satisfacción)
- 📦 Variables operacionales (cantidad, precios)

## 🔍 Uso del Dashboard

### 1️⃣ Carga de Datos
```
• Selecciona "Datos de muestra" para dataset de demostración
• O sube tus propios archivos CSV
• Soporte para múltiples archivos (concatenación automática)
```

### 2️⃣ Exploración Inicial
```
• Revisa métricas básicas del dataset
• Examina la tabla de información de columnas
• Identifica tipos de datos y valores faltantes
```

### 3️⃣ Partición y Filtrado
```
• Selecciona método de partición si es necesario
• Configura parámetros según el método
• Observa cambios en el tamaño del dataset
```

### 4️⃣ Generación de Visualizaciones
```
• Selecciona tipo de gráfico
• Configura columnas y parámetros
• Genera y descarga visualizaciones
```

## ⚡ Rendimiento

- **Tiempo de respuesta**: < 2 segundos para filtros
- **Capacidad**: Hasta 100,000 registros
- **Memoria**: Uso eficiente con particionado automático

## 🌐 Compatibilidad

- **Entrada**: CSV, Excel (XLSX, XLS)
- **Exportación**: PNG (300 DPI), CSV, TXT
- **Navegadores**: Chrome, Firefox, Safari, Edge

## 🎨 Diseño

- ✨ **Layout responsivo** adaptable
- 🎛️ **Panel lateral** para configuración
- 📊 **Área principal** para visualizaciones
- 🎨 **Códigos de color** consistentes

## 🤝 Contribuir

1. Fork el repositorio
2. Crea tu rama de feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Autor

**Tu Nombre**
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu-email@ejemplo.com

## 🙏 Agradecimientos

- [Streamlit](https://streamlit.io/) por el framework de aplicaciones web
- [Plotly](https://plotly.com/) por las visualizaciones interactivas
- [Pandas](https://pandas.pydata.org/) por el procesamiento de datos

---

⭐ **¡Dale una estrella al repositorio si te resultó útil!**
