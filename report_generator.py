import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import save_plot, clear_plots
from fpdf import FPDF

load_dotenv()

class ReportGenerator:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.35,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def generate_full_report(self, df: pd.DataFrame, filename: str):
        clear_plots()
        report_parts = []

        report_parts.append("# AutoEDA Agent - Professional Data Analysis Report\n")
        report_parts.append(f"**Generated:** {datetime.now().strftime('%d %B %Y at %H:%M')}\n")
        report_parts.append(f"**Dataset:** {filename}\n")
        report_parts.append(f"**Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}\n\n")

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # 1. Overview
        report_parts.append("## 1. Dataset Overview\n")
        report_parts.append(df.dtypes.to_frame(name="Data Type").to_markdown() + "\n")

        missing = df.isnull().sum()
        if missing.sum() > 0:
            report_parts.append("### Missing Values\n")
            report_parts.append(missing[missing > 0].to_frame(name="Missing Count").to_markdown() + "\n")

        # 2. Statistical Summary
        report_parts.append("## 2. Statistical Summary\n")
        report_parts.append(df.describe(include='all').round(2).to_markdown() + "\n")

        # 3. Univariate Analysis
        report_parts.append("## 3. Univariate Analysis\n")
        if num_cols:
            report_parts.append("### Numerical Features\n")
            for col in num_cols[:8]:
                fig = plt.figure(figsize=(10, 5))
                sns.histplot(df[col].dropna(), kde=True, color='skyblue')
                plt.title(f'Distribution of {col}')
                path = save_plot(fig, f"hist_{col}.png")
                report_parts.append(f"![{col}]({path})\n")

        if cat_cols:
            report_parts.append("### Categorical Features\n")
            for col in cat_cols[:5]:
                fig = plt.figure(figsize=(10, 5))
                top_categories = df[col].value_counts().head(10)
                sns.barplot(x=top_categories.values, y=top_categories.index, palette='viridis')
                plt.title(f'Top Categories in {col}')
                plt.xlabel('Count')
                plt.xticks(rotation=45)
                path = save_plot(fig, f"count_{col}.png")
                report_parts.append(f"![{col}]({path})\n")

        # 4. Bivariate Analysis
        report_parts.append("## 4. Bivariate Analysis (One vs One)\n")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = plt.figure(figsize=(12, 9))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title("Correlation Heatmap")
            path = save_plot(fig, "correlation_heatmap.png")
            report_parts.append(f"![Correlation Heatmap]({path})\n")

            report_parts.append("### Important Column Comparisons\n")
            corr_pairs = corr.unstack().abs().sort_values(ascending=False)
            corr_pairs = corr_pairs[(corr_pairs < 0.99) & (corr_pairs > 0.25)].drop_duplicates().head(5)

            for (col1, col2), val in corr_pairs.items():
                fig = plt.figure(figsize=(9, 6))
                sns.scatterplot(data=df, x=col1, y=col2, alpha=0.7)
                plt.title(f'{col1} vs {col2} (Correlation = {val:.3f})')
                path = save_plot(fig, f"scatter_{col1}_vs_{col2}.png")
                report_parts.append(f"![{col1} vs {col2}]({path})\n")

        # 5. Insights
        report_parts.append("## 5. Executive Summary & Actionable Insights\n")
        insight_prompt = f"""
        Act as a Senior Data Analyst. Provide professional analysis:

        Dataset Shape: {df.shape}
        Numerical Columns: {num_cols}
        Categorical Columns: {cat_cols}
        Sample Statistics: {df.describe(include='all').round(2).to_string()}

        Give a detailed response with:
        1. Executive Summary (2-3 sentences about the dataset)
        2. 5-7 Key Insights (numbered list with actual observations)
        3. 4-5 Actionable Recommendations (numbered list)

        Be specific and use actual column names and numbers in your response.
        """
        try:
            insights = self.llm.invoke(insight_prompt).content
            report_parts.append(insights)
        except Exception as e:
            report_parts.append(f"Insights generation failed: {str(e)}")

        markdown_content = "\n".join(report_parts)
        md_path = f"reports/{filename}_report.md"
        pdf_path = f"reports/{filename}_report.pdf"

        os.makedirs("reports", exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        self._create_pdf(markdown_content, pdf_path)

        return markdown_content, md_path, pdf_path

    def _create_pdf(self, markdown_text: str, pdf_path: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_margins(15, 15, 15)
        pdf.set_font("Arial", size=10)

        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()

            # Embed images
            if line.startswith('!['):
                match = re.search(r'\!\[.*?\]\((.*?)\)', line)
                if match:
                    img_path = match.group(1)
                    if os.path.exists(img_path):
                        try:
                            if pdf.get_y() > 220:
                                pdf.add_page()
                            pdf.image(img_path, x=15, w=180)
                            pdf.ln(4)
                        except Exception:
                            pass
                continue

            # Skip table lines
            if line.startswith('|') or line.startswith('+-') or line.startswith(':-'):
                continue

            if not line:
                pdf.ln(4)
                continue

            if line.startswith('# '):
                pdf.set_font("Arial", 'B', 15)
                pdf.multi_cell(0, 10, line[2:])
                pdf.set_font("Arial", size=10)
            elif line.startswith('## '):
                pdf.set_font("Arial", 'B', 13)
                pdf.multi_cell(0, 10, line[3:])
                pdf.set_font("Arial", size=10)
            elif line.startswith('### '):
                pdf.set_font("Arial", 'B', 11)
                pdf.multi_cell(0, 8, line[4:])
                pdf.set_font("Arial", size=10)
            else:
                line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                line = re.sub(r'\*(.*?)\*', r'\1', line)
                line = re.sub(r'__(.*?)__', r'\1', line)
                line = re.sub(r'[^\x00-\x7F]+', '', line)
                if line.strip():
                    try:
                        pdf.multi_cell(0, 6, line)
                    except Exception:
                        pass

        pdf.output(pdf_path)
