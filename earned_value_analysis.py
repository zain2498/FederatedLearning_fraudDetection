"""
Earned Value Analysis (EVA) - Academic Software Project Management
Project: Federated Learning for Edge Computing - Fraud Detection
Status Date: 30 November 2025
Methodology: Agile-Scrum
Compliance: PMI Standards

Author: SPM Academic Project
Date: December 2025
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
import os

class EarnedValueAnalysis:
    def __init__(self):
        self.wb = openpyxl.Workbook()
        self.project_title = "Federated Learning for Edge Computing - Fraud Detection"
        self.status_date = "30 November 2025"
        self.methodology = "Agile-Scrum"
        
        # Resource rates (matching budget generator)
        self.resources = {
            "Project Manager": 25.00,
            "Data Scientist 1": 20.00,
            "Data Scientist 2": 20.00,
            "Software Engineer 1": 18.00,
            "Software Engineer 2": 18.00,
            "QA Engineer": 15.00,
            "DevOps Engineer": 22.00
        }
        
        # Tasks to be analyzed (activities through Nov 30, 2025)
        # By Nov 30, we're at end of Month 2 (Oct + Nov)
        self.eva_tasks = self.define_eva_tasks()
        
        # EVA Metrics (to be calculated)
        self.total_pv = 0  # Planned Value (BCWS)
        self.total_ev = 0  # Earned Value (BCWP)
        self.total_ac = 0  # Actual Cost (ACWP)
        self.sv = 0        # Schedule Variance
        self.cv = 0        # Cost Variance
        self.spi = 0       # Schedule Performance Index
        self.cpi = 0       # Cost Performance Index
        self.bac = 65256.00  # Budget at Completion (from previous budget)
        self.etc = 0       # Estimate to Complete
        self.eac = 0       # Estimate at Completion
        
    def define_eva_tasks(self):
        """
        Define tasks with completion status as of Nov 30, 2025
        Rules applied:
        - At least 3 tasks at 100% complete
        - At least 3 tasks at 70% complete
        - At least 3 tasks at 30% complete
        - For completed: AC = PV (except 2 where AC = 2×PV)
        - For in-progress: AC = proportional to % complete
        """
        
        tasks = [
            # ============================================
            # PHASE 1: PLANNING & MANAGEMENT (Month 1 - October)
            # ============================================
            
            # 100% Complete Tasks (3 required minimum)
            {
                "wbs_id": "1.1",
                "task": "Project Initiation & Charter Development",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 40,
                "hourly_rate": 25.00,
                "pv": 40 * 25.00,  # $1,000
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            {
                "wbs_id": "1.2",
                "task": "Stakeholder Analysis & Communication Plan",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 24,
                "hourly_rate": 25.00,
                "pv": 24 * 25.00,  # $600
                "percent_complete": 100,
                "ac_multiplier": 2.0  # AC = 2×PV (cost overrun)
            },
            {
                "wbs_id": "1.3",
                "task": "Risk Management Planning",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 32,
                "hourly_rate": 25.00,
                "pv": 32 * 25.00,  # $800
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            {
                "wbs_id": "1.4",
                "task": "Agile-Scrum Framework Setup",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 16,
                "hourly_rate": 25.00,
                "pv": 16 * 25.00,  # $400
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            
            # ============================================
            # PHASE 2: REQUIREMENTS ANALYSIS (Month 1-2)
            # ============================================
            
            {
                "wbs_id": "2.1",
                "task": "Business Requirements Gathering",
                "phase": "Requirements Analysis",
                "resource": "Project Manager",
                "hours": 40,
                "hourly_rate": 25.00,
                "pv": 40 * 25.00,  # $1,000
                "percent_complete": 100,
                "ac_multiplier": 2.0  # AC = 2×PV (cost overrun - scope clarifications)
            },
            {
                "wbs_id": "2.2",
                "task": "Functional Requirements Documentation",
                "phase": "Requirements Analysis",
                "resource": "Software Engineer 1",
                "hours": 56,
                "hourly_rate": 18.00,
                "pv": 56 * 18.00,  # $1,008
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            {
                "wbs_id": "2.3",
                "task": "Non-Functional Requirements (Performance, Security)",
                "phase": "Requirements Analysis",
                "resource": "Software Engineer 2",
                "hours": 48,
                "hourly_rate": 18.00,
                "pv": 48 * 18.00,  # $864
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            
            # 70% Complete Tasks (3 required minimum)
            {
                "wbs_id": "2.4",
                "task": "Data Privacy & Compliance Analysis (GDPR, PCI-DSS)",
                "phase": "Requirements Analysis",
                "resource": "Data Scientist 1",
                "hours": 40,
                "hourly_rate": 20.00,
                "pv": 40 * 20.00,  # $800
                "percent_complete": 70,
                "ac_multiplier": 0.70  # AC proportional to completion
            },
            {
                "wbs_id": "2.5",
                "task": "Fraud Detection Use Case Analysis",
                "phase": "Requirements Analysis",
                "resource": "Data Scientist 2",
                "hours": 48,
                "hourly_rate": 20.00,
                "pv": 48 * 20.00,  # $960
                "percent_complete": 70,
                "ac_multiplier": 0.70  # AC proportional to completion
            },
            
            # ============================================
            # PHASE 3: SYSTEM & ARCHITECTURE DESIGN (Month 2-3)
            # ============================================
            
            {
                "wbs_id": "3.1",
                "task": "Federated Learning Architecture Design",
                "phase": "System & Architecture Design",
                "resource": "Data Scientist 1",
                "hours": 80,
                "hourly_rate": 20.00,
                "pv": 80 * 20.00,  # $1,600
                "percent_complete": 70,
                "ac_multiplier": 0.70  # AC proportional to completion
            },
            {
                "wbs_id": "3.2",
                "task": "Edge Node Architecture Design",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 1",
                "hours": 72,
                "hourly_rate": 18.00,
                "pv": 72 * 18.00,  # $1,296
                "percent_complete": 70,
                "ac_multiplier": 0.70  # AC proportional to completion
            },
            
            # 30% Complete Tasks (3 required minimum)
            {
                "wbs_id": "3.3",
                "task": "Central Aggregation Server Design",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 2",
                "hours": 64,
                "hourly_rate": 18.00,
                "pv": 64 * 18.00,  # $1,152
                "percent_complete": 30,
                "ac_multiplier": 0.30  # AC proportional to completion
            },
            {
                "wbs_id": "3.4",
                "task": "Communication Protocol Design (gRPC/REST)",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 1",
                "hours": 48,
                "hourly_rate": 18.00,
                "pv": 48 * 18.00,  # $864
                "percent_complete": 30,
                "ac_multiplier": 0.30  # AC proportional to completion
            },
            {
                "wbs_id": "3.5",
                "task": "Database & Storage Architecture",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 2",
                "hours": 40,
                "hourly_rate": 18.00,
                "pv": 40 * 18.00,  # $720
                "percent_complete": 30,
                "ac_multiplier": 0.30  # AC proportional to completion
            },
            {
                "wbs_id": "3.6",
                "task": "Security Architecture & Encryption Design",
                "phase": "System & Architecture Design",
                "resource": "DevOps Engineer",
                "hours": 56,
                "hourly_rate": 22.00,
                "pv": 56 * 22.00,  # $1,232
                "percent_complete": 30,
                "ac_multiplier": 0.30  # AC proportional to completion
            },
            
            # ============================================
            # PROJECT MANAGEMENT ACTIVITIES (Ongoing)
            # ============================================
            
            {
                "wbs_id": "9.1",
                "task": "Sprint Planning & Management (Month 1)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "hourly_rate": 25.00,
                "pv": 32 * 25.00,  # $800
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
            {
                "wbs_id": "9.2",
                "task": "Sprint Planning & Management (Month 2)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "hourly_rate": 25.00,
                "pv": 32 * 25.00,  # $800
                "percent_complete": 100,
                "ac_multiplier": 1.0  # AC = PV
            },
        ]
        
        # Calculate EV and AC for each task
        for task in tasks:
            task["ev"] = task["pv"] * (task["percent_complete"] / 100.0)
            task["ac"] = task["pv"] * task["ac_multiplier"]
        
        return tasks
    
    def calculate_eva_metrics(self):
        """Calculate all EVA metrics using PMI-standard formulas"""
        
        # Sum totals
        self.total_pv = sum(task["pv"] for task in self.eva_tasks)
        self.total_ev = sum(task["ev"] for task in self.eva_tasks)
        self.total_ac = sum(task["ac"] for task in self.eva_tasks)
        
        # Variance calculations
        self.sv = self.total_ev - self.total_pv  # Schedule Variance
        self.cv = self.total_ev - self.total_ac  # Cost Variance
        
        # Performance indices
        self.spi = self.total_ev / self.total_pv if self.total_pv > 0 else 0
        self.cpi = self.total_ev / self.total_ac if self.total_ac > 0 else 0
        
        # Forecasting
        self.eac = self.bac / self.cpi if self.cpi > 0 else self.bac
        self.etc = self.eac - self.total_ac
        
        # Calculate variance at completion
        self.vac = self.bac - self.eac
        
        # Calculate To-Complete Performance Index
        self.tcpi_bac = (self.bac - self.total_ev) / (self.bac - self.total_ac) if (self.bac - self.total_ac) > 0 else 0
        self.tcpi_eac = (self.bac - self.total_ev) / (self.eac - self.total_ac) if (self.eac - self.total_ac) > 0 else 0
    
    def create_task_data_sheet(self):
        """Table 1 - EVA Task Data"""
        if "Sheet" in self.wb.sheetnames:
            self.wb.remove(self.wb["Sheet"])
        
        ws = self.wb.create_sheet("1. EVA Task Data", 0)
        
        # Title
        ws.merge_cells('A1:H1')
        ws['A1'] = f"{self.project_title} - Earned Value Analysis"
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        
        # Status Date
        ws.merge_cells('A2:H2')
        ws['A2'] = f"Status Date: {self.status_date}"
        ws['A2'].font = Font(bold=True, size=11, italic=True)
        ws['A2'].alignment = Alignment(horizontal='center')
        ws['A2'].fill = PatternFill(start_color="E7E6E6", end_color="E7E6E6", fill_type="solid")
        
        # Column headers
        headers = [
            "WBS ID",
            "Task Name",
            "Phase",
            "Planned Value\nPV ($)",
            "% Complete",
            "Earned Value\nEV ($)",
            "Actual Cost\nAC ($)",
            "Status"
        ]
        
        row = 4
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col)
            cell.value = header
            cell.font = Font(bold=True, size=10)
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.font = Font(bold=True, size=10, color="FFFFFF")
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Data rows
        row = 5
        for task in self.eva_tasks:
            ws.cell(row=row, column=1, value=task["wbs_id"])
            ws.cell(row=row, column=2, value=task["task"])
            ws.cell(row=row, column=3, value=task["phase"])
            
            ws.cell(row=row, column=4, value=task["pv"])
            ws.cell(row=row, column=4).number_format = '$#,##0.00'
            
            ws.cell(row=row, column=5, value=task["percent_complete"])
            ws.cell(row=row, column=5).number_format = '0"%"'
            
            ws.cell(row=row, column=6, value=task["ev"])
            ws.cell(row=row, column=6).number_format = '$#,##0.00'
            
            ws.cell(row=row, column=7, value=task["ac"])
            ws.cell(row=row, column=7).number_format = '$#,##0.00'
            
            # Status indicator
            if task["percent_complete"] == 100:
                status = "Completed"
                status_color = "C6E0B4"
            elif task["percent_complete"] >= 70:
                status = "On Track"
                status_color = "FFE699"
            elif task["percent_complete"] >= 30:
                status = "In Progress"
                status_color = "F4B084"
            else:
                status = "Not Started"
                status_color = "FF6B6B"
            
            ws.cell(row=row, column=8, value=status)
            ws.cell(row=row, column=8).fill = PatternFill(start_color=status_color, end_color=status_color, fill_type="solid")
            ws.cell(row=row, column=8).alignment = Alignment(horizontal='center')
            
            # Apply borders
            for col in range(1, 9):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            
            row += 1
        
        # Total row
        ws.cell(row=row, column=1, value="TOTAL")
        ws.cell(row=row, column=1).font = Font(bold=True, size=11)
        ws.merge_cells(f'A{row}:C{row}')
        
        ws.cell(row=row, column=4, value=self.total_pv)
        ws.cell(row=row, column=4).number_format = '$#,##0.00'
        ws.cell(row=row, column=4).font = Font(bold=True)
        ws.cell(row=row, column=4).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        ws.cell(row=row, column=6, value=self.total_ev)
        ws.cell(row=row, column=6).number_format = '$#,##0.00'
        ws.cell(row=row, column=6).font = Font(bold=True)
        ws.cell(row=row, column=6).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        ws.cell(row=row, column=7, value=self.total_ac)
        ws.cell(row=row, column=7).number_format = '$#,##0.00'
        ws.cell(row=row, column=7).font = Font(bold=True)
        ws.cell(row=row, column=7).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        # Column widths
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 48
        ws.column_dimensions['C'].width = 28
        ws.column_dimensions['D'].width = 14
        ws.column_dimensions['E'].width = 12
        ws.column_dimensions['F'].width = 14
        ws.column_dimensions['G'].width = 14
        ws.column_dimensions['H'].width = 14
        
        # Set row heights
        ws.row_dimensions[4].height = 30
    
    def create_eva_summary_sheet(self):
        """Table 2 - EVA Summary with Metrics"""
        ws = self.wb.create_sheet("2. EVA Summary")
        
        # Title
        ws.merge_cells('A1:D1')
        ws['A1'] = f"Earned Value Analysis Summary - {self.status_date}"
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        
        # Project info
        row = 3
        info = [
            ("Project:", self.project_title),
            ("Status Date:", self.status_date),
            ("Methodology:", self.methodology),
            ("Budget at Completion (BAC):", f"${self.bac:,.2f}"),
        ]
        
        for label, value in info:
            ws.cell(row=row, column=1, value=label).font = Font(bold=True)
            ws.merge_cells(f'B{row}:D{row}')
            ws.cell(row=row, column=2, value=value)
            row += 1
        
        row += 1
        
        # Section 1: Primary EVA Metrics
        ws.merge_cells(f'A{row}:D{row}')
        ws[f'A{row}'] = "PRIMARY EVA METRICS"
        ws[f'A{row}'].font = Font(bold=True, size=12, color="FFFFFF")
        ws[f'A{row}'].fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 1
        
        # Headers
        headers = ["Metric", "Formula", "Value", "Interpretation"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        row += 1
        
        # EVA Metrics data
        eva_metrics = [
            {
                "metric": "Planned Value (PV / BCWS)",
                "formula": "Sum of PV for tasks scheduled",
                "value": self.total_pv,
                "interpretation": "Total planned budget for work scheduled through status date"
            },
            {
                "metric": "Earned Value (EV / BCWP)",
                "formula": "Sum of (PV × % Complete)",
                "value": self.total_ev,
                "interpretation": "Value of work actually completed"
            },
            {
                "metric": "Actual Cost (AC / ACWP)",
                "formula": "Sum of actual expenditures",
                "value": self.total_ac,
                "interpretation": "Total actual costs incurred"
            },
        ]
        
        for metric_data in eva_metrics:
            ws.cell(row=row, column=1, value=metric_data["metric"]).font = Font(bold=True)
            ws.cell(row=row, column=2, value=metric_data["formula"])
            ws.cell(row=row, column=3, value=metric_data["value"])
            ws.cell(row=row, column=3).number_format = '$#,##0.00'
            ws.cell(row=row, column=4, value=metric_data["interpretation"])
            ws.cell(row=row, column=4).alignment = Alignment(wrap_text=True)
            
            for col in range(1, 5):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        row += 1
        
        # Section 2: Variance Analysis
        ws.merge_cells(f'A{row}:D{row}')
        ws[f'A{row}'] = "VARIANCE ANALYSIS"
        ws[f'A{row}'].font = Font(bold=True, size=12, color="FFFFFF")
        ws[f'A{row}'].fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 1
        
        # Headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        row += 1
        
        # Schedule variance interpretation
        sv_status = "AHEAD of Schedule" if self.sv > 0 else ("BEHIND Schedule" if self.sv < 0 else "ON Schedule")
        sv_color = "C6E0B4" if self.sv > 0 else ("F4B084" if self.sv < 0 else "FFFFFF")
        
        # Cost variance interpretation
        cv_status = "UNDER Budget" if self.cv > 0 else ("OVER Budget" if self.cv < 0 else "ON Budget")
        cv_color = "C6E0B4" if self.cv > 0 else ("F4B084" if self.cv < 0 else "FFFFFF")
        
        variance_metrics = [
            {
                "metric": "Schedule Variance (SV)",
                "formula": "EV - PV",
                "value": self.sv,
                "interpretation": f"{sv_status}: {'Ahead' if self.sv > 0 else 'Behind'} by ${abs(self.sv):,.2f}",
                "color": sv_color
            },
            {
                "metric": "Cost Variance (CV)",
                "formula": "EV - AC",
                "value": self.cv,
                "interpretation": f"{cv_status}: {'Under' if self.cv > 0 else 'Over'} budget by ${abs(self.cv):,.2f}",
                "color": cv_color
            },
        ]
        
        for metric_data in variance_metrics:
            ws.cell(row=row, column=1, value=metric_data["metric"]).font = Font(bold=True)
            ws.cell(row=row, column=2, value=metric_data["formula"])
            ws.cell(row=row, column=3, value=metric_data["value"])
            ws.cell(row=row, column=3).number_format = '$#,##0.00'
            ws.cell(row=row, column=3).fill = PatternFill(start_color=metric_data["color"], end_color=metric_data["color"], fill_type="solid")
            ws.cell(row=row, column=4, value=metric_data["interpretation"])
            ws.cell(row=row, column=4).alignment = Alignment(wrap_text=True)
            
            for col in range(1, 5):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        row += 1
        
        # Section 3: Performance Indices
        ws.merge_cells(f'A{row}:D{row}')
        ws[f'A{row}'] = "PERFORMANCE INDICES"
        ws[f'A{row}'].font = Font(bold=True, size=12, color="FFFFFF")
        ws[f'A{row}'].fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 1
        
        # Headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        row += 1
        
        # SPI interpretation
        spi_status = "Ahead" if self.spi > 1.0 else ("Behind" if self.spi < 1.0 else "On Track")
        spi_color = "C6E0B4" if self.spi > 1.0 else ("F4B084" if self.spi < 1.0 else "FFFFFF")
        spi_detail = f"Earning ${self.spi:.2f} for every $1.00 planned"
        
        # CPI interpretation
        cpi_status = "Under Budget" if self.cpi > 1.0 else ("Over Budget" if self.cpi < 1.0 else "On Budget")
        cpi_color = "C6E0B4" if self.cpi > 1.0 else ("F4B084" if self.cpi < 1.0 else "FFFFFF")
        cpi_detail = f"Earning ${self.cpi:.2f} for every $1.00 spent"
        
        performance_metrics = [
            {
                "metric": "Schedule Performance Index (SPI)",
                "formula": "EV / PV",
                "value": self.spi,
                "interpretation": f"{spi_status}: {spi_detail}",
                "color": spi_color
            },
            {
                "metric": "Cost Performance Index (CPI)",
                "formula": "EV / AC",
                "value": self.cpi,
                "interpretation": f"{cpi_status}: {cpi_detail}",
                "color": cpi_color
            },
        ]
        
        for metric_data in performance_metrics:
            ws.cell(row=row, column=1, value=metric_data["metric"]).font = Font(bold=True)
            ws.cell(row=row, column=2, value=metric_data["formula"])
            ws.cell(row=row, column=3, value=metric_data["value"])
            ws.cell(row=row, column=3).number_format = '0.00'
            ws.cell(row=row, column=3).fill = PatternFill(start_color=metric_data["color"], end_color=metric_data["color"], fill_type="solid")
            ws.cell(row=row, column=4, value=metric_data["interpretation"])
            ws.cell(row=row, column=4).alignment = Alignment(wrap_text=True)
            
            for col in range(1, 5):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        row += 1
        
        # Section 4: Forecasting Metrics
        ws.merge_cells(f'A{row}:D{row}')
        ws[f'A{row}'] = "FORECASTING METRICS"
        ws[f'A{row}'].font = Font(bold=True, size=12, color="FFFFFF")
        ws[f'A{row}'].fill = PatternFill(start_color="C55A11", end_color="C55A11", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 1
        
        # Headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="F4B084", end_color="F4B084", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        row += 1
        
        forecasting_metrics = [
            {
                "metric": "Estimate at Completion (EAC)",
                "formula": "BAC / CPI",
                "value": self.eac,
                "interpretation": f"Projected total cost at project completion (Original: ${self.bac:,.2f})",
                "color": "F4B084" if self.eac > self.bac else "C6E0B4"
            },
            {
                "metric": "Estimate to Complete (ETC)",
                "formula": "EAC - AC",
                "value": self.etc,
                "interpretation": "Additional funds required to complete the project",
                "color": "FFFFFF"
            },
            {
                "metric": "Variance at Completion (VAC)",
                "formula": "BAC - EAC",
                "value": self.vac,
                "interpretation": f"Expected {'surplus' if self.vac > 0 else 'overrun'} at completion",
                "color": "C6E0B4" if self.vac > 0 else "F4B084"
            },
            {
                "metric": "To-Complete Performance Index (TCPI-BAC)",
                "formula": "(BAC - EV) / (BAC - AC)",
                "value": self.tcpi_bac,
                "interpretation": f"Performance efficiency needed: {self.tcpi_bac:.2f} to meet original budget",
                "color": "C6E0B4" if self.tcpi_bac < 1.0 else "F4B084"
            },
        ]
        
        for metric_data in forecasting_metrics:
            ws.cell(row=row, column=1, value=metric_data["metric"]).font = Font(bold=True)
            ws.cell(row=row, column=2, value=metric_data["formula"])
            ws.cell(row=row, column=3, value=metric_data["value"])
            
            if "Index" in metric_data["metric"]:
                ws.cell(row=row, column=3).number_format = '0.00'
            else:
                ws.cell(row=row, column=3).number_format = '$#,##0.00'
            
            ws.cell(row=row, column=3).fill = PatternFill(start_color=metric_data["color"], end_color=metric_data["color"], fill_type="solid")
            ws.cell(row=row, column=4, value=metric_data["interpretation"])
            ws.cell(row=row, column=4).alignment = Alignment(wrap_text=True)
            
            for col in range(1, 5):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        # Column widths
        ws.column_dimensions['A'].width = 35
        ws.column_dimensions['B'].width = 22
        ws.column_dimensions['C'].width = 15
        ws.column_dimensions['D'].width = 50
    
    def create_management_summary_sheet(self):
        """Sheet 3 - Management Summary & Recommendations"""
        ws = self.wb.create_sheet("3. Management Summary")
        
        # Title
        ws.merge_cells('A1:B1')
        ws['A1'] = "EARNED VALUE ANALYSIS - MANAGEMENT SUMMARY"
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        
        row = 3
        
        # Executive Summary
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "EXECUTIVE SUMMARY"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        row += 1
        
        exec_summary = f"""As of {self.status_date}, the {self.project_title} has completed 2 months of its 8-month lifecycle.

KEY FINDINGS:

1. SCHEDULE PERFORMANCE:
   • Schedule Performance Index (SPI): {self.spi:.2f}
   • Status: {"AHEAD of schedule" if self.spi > 1.0 else "BEHIND schedule" if self.spi < 1.0 else "ON schedule"}
   • Schedule Variance: ${self.sv:,.2f}
   • The project is currently {"earning more value than planned" if self.spi > 1.0 else "not meeting planned progress"}

2. COST PERFORMANCE:
   • Cost Performance Index (CPI): {self.cpi:.2f}
   • Status: {"UNDER budget" if self.cpi > 1.0 else "OVER budget" if self.cpi < 1.0 else "ON budget"}
   • Cost Variance: ${self.cv:,.2f}
   • The project is {"performing efficiently with costs" if self.cpi > 1.0 else "experiencing cost overruns"}

3. BUDGET PROJECTION:
   • Original Budget (BAC): ${self.bac:,.2f}
   • Projected Final Cost (EAC): ${self.eac:,.2f}
   • Expected Variance: ${self.vac:,.2f} ({"surplus" if self.vac > 0 else "overrun"})
   • Remaining Budget Required (ETC): ${self.etc:,.2f}
"""
        
        ws.merge_cells(f'A{row}:B{row+15}')
        cell = ws[f'A{row}']
        cell.value = exec_summary
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        row += 16
        
        # Risk Assessment
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "RISK ASSESSMENT"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        row += 1
        
        # Determine overall risk level
        if self.cpi < 0.90 or self.spi < 0.90:
            risk_level = "HIGH RISK"
            risk_color = "FF6B6B"
        elif self.cpi < 0.95 or self.spi < 0.95:
            risk_level = "MEDIUM RISK"
            risk_color = "FFE699"
        else:
            risk_level = "LOW RISK"
            risk_color = "C6E0B4"
        
        risk_assessment = f"""Overall Project Risk Level: {risk_level}

IDENTIFIED CONCERNS:
"""
        
        if self.cpi < 1.0:
            risk_assessment += f"\n⚠ Cost Overruns Detected: Two tasks experienced actual costs at 2× planned budget"
            risk_assessment += f"\n   - Stakeholder Analysis: ${600.00:,.2f} over budget"
            risk_assessment += f"\n   - Business Requirements: ${1000.00:,.2f} over budget"
        
        if self.tcpi_bac > 1.1:
            risk_assessment += f"\n⚠ To-Complete Performance Index (TCPI): {self.tcpi_bac:.2f}"
            risk_assessment += f"\n   - Team must improve efficiency by {((self.tcpi_bac - 1.0) * 100):.1f}% to meet original budget"
        
        ws.merge_cells(f'A{row}:B{row+8}')
        cell = ws[f'A{row}']
        cell.value = risk_assessment
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        cell.fill = PatternFill(start_color=risk_color, end_color=risk_color, fill_type="solid")
        row += 9
        
        # Recommendations
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "MANAGEMENT RECOMMENDATIONS"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        row += 1
        
        recommendations = """IMMEDIATE ACTIONS REQUIRED:

1. COST CONTROL:
   ✓ Investigate root causes of cost overruns in planning phase
   ✓ Implement stricter change control procedures
   ✓ Review resource allocation for remaining phases
   ✓ Consider negotiating vendor contracts for upcoming procurement

2. SCHEDULE OPTIMIZATION:
   ✓ Leverage current schedule advantage to build buffer
   ✓ Identify opportunities to fast-track critical path activities
   ✓ Maintain current momentum in requirements and design phases

3. QUALITY ASSURANCE:
   ✓ Ensure quality is not compromised due to cost pressures
   ✓ Implement rigorous code review processes early
   ✓ Allocate sufficient time for testing phases

4. STAKEHOLDER COMMUNICATION:
   ✓ Present EVA findings in next steering committee meeting
   ✓ Discuss budget re-baseline options if CPI trend continues
   ✓ Set realistic expectations for project completion cost

5. RISK MITIGATION:
   ✓ Develop contingency plans for Model Development phase (highest risk)
   ✓ Establish reserve fund for unforeseen technical challenges
   ✓ Monitor CPI and SPI trends weekly
"""
        
        ws.merge_cells(f'A{row}:B{row+22}')
        cell = ws[f'A{row}']
        cell.value = recommendations
        cell.alignment = Alignment(wrap_text=True, vertical='top')
        row += 23
        
        # Project Health Dashboard
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "PROJECT HEALTH DASHBOARD"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        row += 1
        
        # Health indicators
        health_metrics = [
            ("Schedule Health", "On Track" if self.spi >= 0.95 else "At Risk", "C6E0B4" if self.spi >= 0.95 else "FFE699"),
            ("Cost Health", "On Track" if self.cpi >= 0.95 else "At Risk", "C6E0B4" if self.cpi >= 0.95 else "FFE699"),
            ("Overall Progress", f"{(self.total_ev / self.bac * 100):.1f}% Complete", "FFFFFF"),
            ("Budget Consumed", f"{(self.total_ac / self.bac * 100):.1f}%", "FFFFFF"),
        ]
        
        for metric_name, metric_value, color in health_metrics:
            ws.cell(row=row, column=1, value=metric_name).font = Font(bold=True)
            ws.cell(row=row, column=2, value=metric_value)
            ws.cell(row=row, column=2).fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
            row += 1
        
        # Column widths
        ws.column_dimensions['A'].width = 40
        ws.column_dimensions['B'].width = 60
    
    def generate_eva_report(self, output_file="EVA_Report_Federated_Learning.xlsx"):
        """Generate complete EVA report"""
        
        print("=" * 80)
        print("EARNED VALUE ANALYSIS (EVA) GENERATOR")
        print("=" * 80)
        print(f"Project: {self.project_title}")
        print(f"Status Date: {self.status_date}")
        print(f"Methodology: {self.methodology}")
        print("Compliance: PMI Standards")
        print("=" * 80)
        
        print("\n[1/5] Calculating EVA metrics...")
        self.calculate_eva_metrics()
        
        print("[2/5] Generating EVA Task Data sheet...")
        self.create_task_data_sheet()
        
        print("[3/5] Generating EVA Summary sheet...")
        self.create_eva_summary_sheet()
        
        print("[4/5] Generating Management Summary...")
        self.create_management_summary_sheet()
        
        print("[5/5] Saving Excel report...")
        output_path = os.path.join(os.getcwd(), output_file)
        self.wb.save(output_path)
        
        print("\n" + "=" * 80)
        print("EARNED VALUE ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\n📊 EVA RESULTS SUMMARY:")
        print(f"\n   PRIMARY METRICS:")
        print(f"   ├─ Planned Value (PV):     ${self.total_pv:>12,.2f}")
        print(f"   ├─ Earned Value (EV):      ${self.total_ev:>12,.2f}")
        print(f"   └─ Actual Cost (AC):       ${self.total_ac:>12,.2f}")
        
        print(f"\n   VARIANCE ANALYSIS:")
        print(f"   ├─ Schedule Variance (SV): ${self.sv:>12,.2f}  {'✓ AHEAD' if self.sv > 0 else '✗ BEHIND' if self.sv < 0 else '• ON TRACK'}")
        print(f"   └─ Cost Variance (CV):     ${self.cv:>12,.2f}  {'✓ UNDER' if self.cv > 0 else '✗ OVER' if self.cv < 0 else '• ON BUDGET'}")
        
        print(f"\n   PERFORMANCE INDICES:")
        print(f"   ├─ SPI (Schedule):         {self.spi:>12.2f}  {'✓ Ahead' if self.spi > 1.0 else '✗ Behind' if self.spi < 1.0 else '• On Track'}")
        print(f"   └─ CPI (Cost):             {self.cpi:>12.2f}  {'✓ Efficient' if self.cpi > 1.0 else '✗ Overrun' if self.cpi < 1.0 else '• On Budget'}")
        
        print(f"\n   FORECASTING:")
        print(f"   ├─ Budget at Completion:   ${self.bac:>12,.2f}")
        print(f"   ├─ Estimate at Completion: ${self.eac:>12,.2f}")
        print(f"   ├─ Variance at Completion: ${self.vac:>12,.2f}")
        print(f"   └─ Estimate to Complete:   ${self.etc:>12,.2f}")
        
        print(f"\n   PROJECT HEALTH:")
        health_status = "✓ HEALTHY" if (self.cpi >= 0.95 and self.spi >= 0.95) else "⚠ AT RISK" if (self.cpi >= 0.90 and self.spi >= 0.90) else "✗ CRITICAL"
        print(f"   └─ Overall Status:         {health_status}")
        
        print(f"\n📁 File saved: {output_path}")
        print(f"📝 Tasks Analyzed: {len(self.eva_tasks)}")
        print(f"   ├─ Completed (100%): {sum(1 for t in self.eva_tasks if t['percent_complete'] == 100)}")
        print(f"   ├─ On Track (70%):   {sum(1 for t in self.eva_tasks if t['percent_complete'] == 70)}")
        print(f"   └─ In Progress (30%): {sum(1 for t in self.eva_tasks if t['percent_complete'] == 30)}")
        
        print("\n" + "=" * 80)
        print("✓ Report suitable for: SPM Term Project | Viva Defense | Expert Validation")
        print("=" * 80)
        
        return output_path

if __name__ == "__main__":
    eva = EarnedValueAnalysis()
    eva.generate_eva_report()
