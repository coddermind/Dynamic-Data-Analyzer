# Dynamic Data Analysis Platform

A comprehensive web-based data analysis and visualization platform built with Django, designed to provide powerful data preprocessing, analysis, and visualization capabilities through an intuitive web interface.

## 🌟 Features

### 📊 Data Management
- **Multi-format Support**: Upload and process CSV and Excel files with automatic format detection
- **User Authentication**: Secure user registration and login system with personal dataset management
- **Private Datasets**: Each user has access only to their own uploaded datasets
- **Dataset Metadata**: Automatic extraction of dataset shape, column information, and data types

### 🔧 Advanced Data Preprocessing
- **Missing Value Handling**: Multiple strategies including drop, mean/median/mode imputation, and custom fill values
- **Feature Encoding**: One-hot encoding, label encoding for categorical variables
- **Data Scaling**: Min-Max, Standard (Z-score), and Robust scaling options
- **Outlier Detection & Treatment**: Automatic outlier detection with capping or removal options
- **Feature Selection**: Variance threshold and K-best feature selection methods
- **Dimensionality Reduction**: Principal Component Analysis (PCA) integration
- **Column-Specific Processing**: Granular control over preprocessing at the column level
- **Data Filtering**: Advanced conditional filtering for numeric, categorical, and datetime columns

### 📈 Interactive Visualizations
- **Single Column Analysis**: Histograms, box plots, value counts for individual columns
- **Comparative Analysis**: Scatter plots, correlation matrices, pair plots for multi-column analysis
- **Advanced Visualizations**: Heatmaps, 3D scatter plots, distribution comparisons
- **Interactive Charts**: Powered by Plotly for dynamic, interactive data exploration
- **Export Capabilities**: Download visualizations and processed datasets

### 🎨 User Experience
- **Modern UI**: Bootstrap-based responsive design with Font Awesome icons
- **Real-time Feedback**: AJAX-powered interactions for seamless user experience
- **Progress Tracking**: Visual feedback during data processing operations
- **Data Preview**: Live preview of datasets before and after preprocessing
- **Column Statistics**: Comprehensive statistical summaries for all columns

## 🛠️ Technical Stack

- **Backend**: Django 3.2.23 (Python web framework)
- **Data Processing**: 
  - Pandas 2.0.3 (Data manipulation and analysis)
  - NumPy 1.24.4 (Numerical computing)
  - Scikit-learn 1.2.2 (Machine learning preprocessing)
- **Visualization**: 
  - Plotly 5.20.0 (Interactive web visualizations)
  - Matplotlib 3.7.3 (Static plotting)
  - Seaborn 0.13.2 (Statistical data visualization)
- **File Processing**: OpenPyXL 3.1.2 (Excel file handling)
- **Database**: SQLite (Development), PostgreSQL-ready for production
- **Frontend**: Bootstrap 5, jQuery, Font Awesome

## 📋 Prerequisites

- Python 3.8+ 
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dynamic-data-analysis
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux  
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
```bash
python manage.py migrate
```

### 5. Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 6. Run Development Server
```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000/`

## 📖 Usage Guide

### Getting Started
1. **Register/Login**: Create an account or login to access the platform
2. **Upload Dataset**: Navigate to "Upload Dataset" and select your CSV or Excel file
3. **Preprocess Data**: Configure preprocessing options including:
   - Missing value strategies
   - Feature encoding methods
   - Scaling techniques
   - Outlier handling
   - Column-specific settings
4. **Visualize**: Explore your data with various visualization options
5. **Export**: Download processed datasets or visualization files

### Data Upload
- Supported formats: CSV, Excel (.xls, .xlsx)
- Maximum file size: 10MB (configurable)
- Automatic encoding detection for CSV files
- Column type inference (numeric, categorical, datetime, text, binary)

### Preprocessing Options

#### Global Settings
- **Missing Values**: Drop rows, fill with statistical measures, or custom values
- **Encoding**: One-hot or label encoding for categorical variables
- **Scaling**: Standardization, normalization, or robust scaling
- **Feature Selection**: Remove low-variance features or select top K features
- **Dimensionality Reduction**: PCA with configurable components

#### Column-Specific Settings
- Override global settings for individual columns
- Custom conditional filtering
- Value replacement rules
- Column-specific outlier handling

### Visualization Types

#### Single Column Analysis
- **Numeric**: Histograms, box plots, distribution curves
- **Categorical**: Bar charts, pie charts, value counts
- **DateTime**: Time series plots, seasonal decomposition

#### Multi-Column Analysis
- **Correlation**: Heatmaps, correlation matrices
- **Comparison**: Scatter plots, pair plots
- **Distribution**: Overlaid histograms, violin plots

#### Advanced Visualizations
- **3D Scatter Plots**: Multi-dimensional data exploration
- **Interactive Dashboards**: Linked charts and filters
- **Custom Plots**: Configurable chart types and styling

## 🏗️ Project Structure

```
dynamic-data-analysis/
├── analyzer/                 # Main Django app
│   ├── models.py            # Database models (Dataset, Preprocessing)
│   ├── views.py             # Application logic and API endpoints
│   ├── utils.py             # Data processing utilities
│   ├── urls.py              # URL routing
│   └── admin.py             # Django admin configuration
├── data_analyzer/           # Django project settings
│   ├── settings.py          # Configuration
│   ├── urls.py              # Main URL routing
│   └── wsgi.py              # WSGI application
├── templates/               # HTML templates
│   ├── analyzer/            # App-specific templates
│   └── registration/        # Authentication templates
├── static/                  # Static files (CSS, JS, images)
├── media/                   # User uploaded files
│   └── datasets/            # Dataset storage
├── manage.py                # Django management script
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for production settings:
```env
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=your-domain.com
DATABASE_URL=your-database-url
```

### File Upload Settings
Modify `settings.py` to adjust file upload limits:
```python
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10MB
```

## 🚢 Deployment

### Production Checklist
- [ ] Set `DEBUG = False` in settings
- [ ] Configure `ALLOWED_HOSTS`
- [ ] Use PostgreSQL or MySQL for production database
- [ ] Set up static file serving (WhiteNoise or CDN)
- [ ] Configure media file storage (AWS S3, etc.)
- [ ] Set up SSL/HTTPS
- [ ] Configure logging
- [ ] Set up monitoring and error tracking

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "data_analyzer.wsgi"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Django**: Web framework
- **Pandas**: Data manipulation library
- **Plotly**: Interactive visualization library
- **Scikit-learn**: Machine learning preprocessing tools
- **Bootstrap**: Frontend CSS framework

## 📞 Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for detailed API references

---

**Built with ❤️ for data enthusiasts and analysts**  
