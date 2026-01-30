"""
Time-Phased Project Budget Generator
Project: Federated Learning for Edge Computing for Fraud Detection
Duration: 8 months (October 2025 – May 2026)
Methodology: Agile–Scrum
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from datetime import datetime
import os

class ProjectBudgetGenerator:
    def __init__(self):
        self.wb = openpyxl.Workbook()
        self.project_title = "Federated Learning for Edge Computing - Fraud Detection"
        self.duration_months = 8
        self.months = ["October 2025", "November 2025", "December 2025", 
                      "January 2026", "February 2026", "March 2026", 
                      "April 2026", "May 2026"]
        
        # Pakistani market rates (USD) - Academic/Industry mix
        self.resources = {
            "Project Manager": {
                "count": 1,
                "hourly_rate": 25.00,  # $25/hr realistic for Pakistan senior role
                "total_hours": 0
            },
            "Data Scientist 1": {
                "count": 1,
                "hourly_rate": 20.00,  # $20/hr
                "total_hours": 0
            },
            "Data Scientist 2": {
                "count": 1,
                "hourly_rate": 20.00,
                "total_hours": 0
            },
            "Software Engineer 1": {
                "count": 1,
                "hourly_rate": 18.00,  # $18/hr
                "total_hours": 0
            },
            "Software Engineer 2": {
                "count": 1,
                "hourly_rate": 18.00,
                "total_hours": 0
            },
            "QA Engineer": {
                "count": 1,
                "hourly_rate": 15.00,  # $15/hr
                "total_hours": 0
            },
            "DevOps Engineer": {
                "count": 1,
                "hourly_rate": 22.00,  # $22/hr
                "total_hours": 0
            }
        }
        
        # Working hours: 8 hrs/day, 5 days/week ≈ 160 hrs/month
        self.hours_per_month = 160
        
        # Overhead rate
        self.overhead_rate = 0.30  # 30%
        
        # WBS Tasks with detailed breakdown
        self.wbs_tasks = self.define_wbs_tasks()
        
    def define_wbs_tasks(self):
        """Define comprehensive Work Breakdown Structure"""
        return [
            # Phase 1: Project Planning & Management (Month 1 - Oct 2025)
            {
                "wbs_id": "1.1",
                "task": "Project Initiation & Charter Development",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 40,
                "month": "October 2025"
            },
            {
                "wbs_id": "1.2",
                "task": "Stakeholder Analysis & Communication Plan",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 24,
                "month": "October 2025"
            },
            {
                "wbs_id": "1.3",
                "task": "Risk Management Planning",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "October 2025"
            },
            {
                "wbs_id": "1.4",
                "task": "Agile-Scrum Framework Setup",
                "phase": "Planning & Management",
                "resource": "Project Manager",
                "hours": 16,
                "month": "October 2025"
            },
            
            # Phase 2: Requirements Analysis (Month 1-2 - Oct-Nov 2025)
            {
                "wbs_id": "2.1",
                "task": "Business Requirements Gathering",
                "phase": "Requirements Analysis",
                "resource": "Project Manager",
                "hours": 40,
                "month": "October 2025"
            },
            {
                "wbs_id": "2.2",
                "task": "Functional Requirements Documentation",
                "phase": "Requirements Analysis",
                "resource": "Software Engineer 1",
                "hours": 56,
                "month": "October 2025"
            },
            {
                "wbs_id": "2.3",
                "task": "Non-Functional Requirements (Performance, Security)",
                "phase": "Requirements Analysis",
                "resource": "Software Engineer 2",
                "hours": 48,
                "month": "October 2025"
            },
            {
                "wbs_id": "2.4",
                "task": "Data Privacy & Compliance Analysis (GDPR, PCI-DSS)",
                "phase": "Requirements Analysis",
                "resource": "Data Scientist 1",
                "hours": 40,
                "month": "November 2025"
            },
            {
                "wbs_id": "2.5",
                "task": "Fraud Detection Use Case Analysis",
                "phase": "Requirements Analysis",
                "resource": "Data Scientist 2",
                "hours": 48,
                "month": "November 2025"
            },
            
            # Phase 3: System & Architecture Design (Month 2-3 - Nov-Dec 2025)
            {
                "wbs_id": "3.1",
                "task": "Federated Learning Architecture Design",
                "phase": "System & Architecture Design",
                "resource": "Data Scientist 1",
                "hours": 80,
                "month": "November 2025"
            },
            {
                "wbs_id": "3.2",
                "task": "Edge Node Architecture Design",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 1",
                "hours": 72,
                "month": "November 2025"
            },
            {
                "wbs_id": "3.3",
                "task": "Central Aggregation Server Design",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 2",
                "hours": 64,
                "month": "November 2025"
            },
            {
                "wbs_id": "3.4",
                "task": "Communication Protocol Design (gRPC/REST)",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 1",
                "hours": 48,
                "month": "December 2025"
            },
            {
                "wbs_id": "3.5",
                "task": "Database & Storage Architecture",
                "phase": "System & Architecture Design",
                "resource": "Software Engineer 2",
                "hours": 40,
                "month": "December 2025"
            },
            {
                "wbs_id": "3.6",
                "task": "Security Architecture & Encryption Design",
                "phase": "System & Architecture Design",
                "resource": "DevOps Engineer",
                "hours": 56,
                "month": "December 2025"
            },
            {
                "wbs_id": "3.7",
                "task": "Infrastructure & Deployment Architecture",
                "phase": "System & Architecture Design",
                "resource": "DevOps Engineer",
                "hours": 48,
                "month": "December 2025"
            },
            
            # Phase 4: Data Preprocessing & Feature Engineering (Month 3-4 - Dec 2025-Jan 2026)
            {
                "wbs_id": "4.1",
                "task": "Credit Card Dataset Acquisition & Preparation",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 1",
                "hours": 56,
                "month": "December 2025"
            },
            {
                "wbs_id": "4.2",
                "task": "Data Cleaning & Missing Value Handling",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 2",
                "hours": 64,
                "month": "December 2025"
            },
            {
                "wbs_id": "4.3",
                "task": "Feature Engineering & Selection",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 1",
                "hours": 72,
                "month": "January 2026"
            },
            {
                "wbs_id": "4.4",
                "task": "Data Partitioning for Federated Nodes",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 2",
                "hours": 56,
                "month": "January 2026"
            },
            {
                "wbs_id": "4.5",
                "task": "Imbalanced Data Handling (SMOTE/ADASYN)",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 1",
                "hours": 48,
                "month": "January 2026"
            },
            {
                "wbs_id": "4.6",
                "task": "Data Privacy Preservation (Differential Privacy)",
                "phase": "Data Preprocessing",
                "resource": "Data Scientist 2",
                "hours": 40,
                "month": "January 2026"
            },
            
            # Phase 5: Model Development - Federated Learning (Month 4-5 - Jan-Feb 2026)
            {
                "wbs_id": "5.1",
                "task": "Base Model Selection & Configuration (MLP/XGBoost)",
                "phase": "Model Development",
                "resource": "Data Scientist 1",
                "hours": 64,
                "month": "January 2026"
            },
            {
                "wbs_id": "5.2",
                "task": "Local Model Training Implementation",
                "phase": "Model Development",
                "resource": "Data Scientist 1",
                "hours": 80,
                "month": "February 2026"
            },
            {
                "wbs_id": "5.3",
                "task": "Federated Averaging (FedAvg) Algorithm Implementation",
                "phase": "Model Development",
                "resource": "Data Scientist 2",
                "hours": 88,
                "month": "February 2026"
            },
            {
                "wbs_id": "5.4",
                "task": "Hierarchical Federated Learning Implementation",
                "phase": "Model Development",
                "resource": "Data Scientist 2",
                "hours": 72,
                "month": "February 2026"
            },
            {
                "wbs_id": "5.5",
                "task": "Model Aggregation Server Development",
                "phase": "Model Development",
                "resource": "Software Engineer 1",
                "hours": 96,
                "month": "February 2026"
            },
            {
                "wbs_id": "5.6",
                "task": "Edge Client Application Development",
                "phase": "Model Development",
                "resource": "Software Engineer 2",
                "hours": 104,
                "month": "February 2026"
            },
            
            # Phase 6: Edge Deployment & Integration (Month 5-6 - Feb-Mar 2026)
            {
                "wbs_id": "6.1",
                "task": "Edge Node Setup & Configuration",
                "phase": "Edge Deployment",
                "resource": "DevOps Engineer",
                "hours": 80,
                "month": "February 2026"
            },
            {
                "wbs_id": "6.2",
                "task": "Containerization (Docker/Kubernetes)",
                "phase": "Edge Deployment",
                "resource": "DevOps Engineer",
                "hours": 72,
                "month": "March 2026"
            },
            {
                "wbs_id": "6.3",
                "task": "CI/CD Pipeline Setup",
                "phase": "Edge Deployment",
                "resource": "DevOps Engineer",
                "hours": 64,
                "month": "March 2026"
            },
            {
                "wbs_id": "6.4",
                "task": "Network Configuration & Load Balancing",
                "phase": "Edge Deployment",
                "resource": "DevOps Engineer",
                "hours": 48,
                "month": "March 2026"
            },
            {
                "wbs_id": "6.5",
                "task": "Monitoring & Logging Setup (Prometheus/Grafana)",
                "phase": "Edge Deployment",
                "resource": "DevOps Engineer",
                "hours": 56,
                "month": "March 2026"
            },
            {
                "wbs_id": "6.6",
                "task": "API Gateway & Service Mesh Implementation",
                "phase": "Edge Deployment",
                "resource": "Software Engineer 1",
                "hours": 64,
                "month": "March 2026"
            },
            {
                "wbs_id": "6.7",
                "task": "Database Integration & Optimization",
                "phase": "Edge Deployment",
                "resource": "Software Engineer 2",
                "hours": 56,
                "month": "March 2026"
            },
            
            # Phase 7: Testing & QA (Month 6-7 - Mar-Apr 2026)
            {
                "wbs_id": "7.1",
                "task": "Unit Testing - Model Components",
                "phase": "Testing & QA",
                "resource": "QA Engineer",
                "hours": 72,
                "month": "March 2026"
            },
            {
                "wbs_id": "7.2",
                "task": "Integration Testing - Federated System",
                "phase": "Testing & QA",
                "resource": "QA Engineer",
                "hours": 88,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.3",
                "task": "Performance Testing & Benchmarking",
                "phase": "Testing & QA",
                "resource": "QA Engineer",
                "hours": 64,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.4",
                "task": "Security Testing & Penetration Testing",
                "phase": "Testing & QA",
                "resource": "QA Engineer",
                "hours": 56,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.5",
                "task": "Privacy Validation & Compliance Testing",
                "phase": "Testing & QA",
                "resource": "Data Scientist 1",
                "hours": 48,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.6",
                "task": "Model Performance Evaluation (Accuracy, F1-Score)",
                "phase": "Testing & QA",
                "resource": "Data Scientist 2",
                "hours": 56,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.7",
                "task": "User Acceptance Testing (UAT)",
                "phase": "Testing & QA",
                "resource": "Project Manager",
                "hours": 40,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.8",
                "task": "Bug Fixing & Issue Resolution",
                "phase": "Testing & QA",
                "resource": "Software Engineer 1",
                "hours": 64,
                "month": "April 2026"
            },
            {
                "wbs_id": "7.9",
                "task": "Regression Testing",
                "phase": "Testing & QA",
                "resource": "QA Engineer",
                "hours": 48,
                "month": "April 2026"
            },
            
            # Phase 8: Documentation & Final Reporting (Month 7-8 - Apr-May 2026)
            {
                "wbs_id": "8.1",
                "task": "Technical Documentation - Architecture",
                "phase": "Documentation & Reporting",
                "resource": "Software Engineer 1",
                "hours": 56,
                "month": "April 2026"
            },
            {
                "wbs_id": "8.2",
                "task": "API Documentation & Developer Guide",
                "phase": "Documentation & Reporting",
                "resource": "Software Engineer 2",
                "hours": 48,
                "month": "April 2026"
            },
            {
                "wbs_id": "8.3",
                "task": "User Manual & Deployment Guide",
                "phase": "Documentation & Reporting",
                "resource": "DevOps Engineer",
                "hours": 40,
                "month": "May 2026"
            },
            {
                "wbs_id": "8.4",
                "task": "Model Documentation & Research Paper",
                "phase": "Documentation & Reporting",
                "resource": "Data Scientist 1",
                "hours": 64,
                "month": "May 2026"
            },
            {
                "wbs_id": "8.5",
                "task": "Experiment Results & Performance Analysis Report",
                "phase": "Documentation & Reporting",
                "resource": "Data Scientist 2",
                "hours": 56,
                "month": "May 2026"
            },
            {
                "wbs_id": "8.6",
                "task": "Project Final Report & Lessons Learned",
                "phase": "Documentation & Reporting",
                "resource": "Project Manager",
                "hours": 72,
                "month": "May 2026"
            },
            {
                "wbs_id": "8.7",
                "task": "Presentation & Stakeholder Demo",
                "phase": "Documentation & Reporting",
                "resource": "Project Manager",
                "hours": 32,
                "month": "May 2026"
            },
            {
                "wbs_id": "8.8",
                "task": "Knowledge Transfer & Training",
                "phase": "Documentation & Reporting",
                "resource": "Project Manager",
                "hours": 40,
                "month": "May 2026"
            },
            
            # Ongoing Project Management Activities (Monthly)
            {
                "wbs_id": "9.1",
                "task": "Sprint Planning & Management (Month 1)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "October 2025"
            },
            {
                "wbs_id": "9.2",
                "task": "Sprint Planning & Management (Month 2)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "November 2025"
            },
            {
                "wbs_id": "9.3",
                "task": "Sprint Planning & Management (Month 3)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "December 2025"
            },
            {
                "wbs_id": "9.4",
                "task": "Sprint Planning & Management (Month 4)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "January 2026"
            },
            {
                "wbs_id": "9.5",
                "task": "Sprint Planning & Management (Month 5)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "February 2026"
            },
            {
                "wbs_id": "9.6",
                "task": "Sprint Planning & Management (Month 6)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "March 2026"
            },
            {
                "wbs_id": "9.7",
                "task": "Sprint Planning & Management (Month 7)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "April 2026"
            },
            {
                "wbs_id": "9.8",
                "task": "Sprint Planning & Management (Month 8)",
                "phase": "Project Management",
                "resource": "Project Manager",
                "hours": 32,
                "month": "May 2026"
            },
        ]
    
    def calculate_resource_totals(self):
        """Calculate total hours per resource"""
        for task in self.wbs_tasks:
            resource = task["resource"]
            hours = task["hours"]
            if resource in self.resources:
                self.resources[resource]["total_hours"] += hours
    
    def create_resource_sheet(self):
        """Sheet 1: Resource Sheet"""
        if "Sheet" in self.wb.sheetnames:
            self.wb.remove(self.wb["Sheet"])
        
        ws = self.wb.create_sheet("1. Resource Sheet", 0)
        
        # Header
        ws.merge_cells('A1:E1')
        ws['A1'] = f"{self.project_title} - Resource Budget"
        ws['A1'].font = Font(bold=True, size=14)
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        
        # Column headers
        headers = ["Resource Name", "Role", "Hourly Rate (USD)", "Total Allocated Hours", "Total Cost (USD)"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col)
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
        
        # Data rows
        row = 4
        total_cost = 0
        for resource_name, details in self.resources.items():
            ws.cell(row=row, column=1, value=resource_name)
            ws.cell(row=row, column=2, value=resource_name.split()[0] + " " + resource_name.split()[-1])
            ws.cell(row=row, column=3, value=details["hourly_rate"])
            ws.cell(row=row, column=3).number_format = '$#,##0.00'
            ws.cell(row=row, column=4, value=details["total_hours"])
            
            cost = details["hourly_rate"] * details["total_hours"]
            ws.cell(row=row, column=5, value=cost)
            ws.cell(row=row, column=5).number_format = '$#,##0.00'
            total_cost += cost
            
            # Apply borders
            for col in range(1, 6):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        # Total row
        ws.cell(row=row, column=1, value="TOTAL")
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=5, value=total_cost)
        ws.cell(row=row, column=5).number_format = '$#,##0.00'
        ws.cell(row=row, column=5).font = Font(bold=True)
        ws.cell(row=row, column=5).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        # Column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20
        ws.column_dimensions['D'].width = 22
        ws.column_dimensions['E'].width = 20
        
        return total_cost
    
    def create_cost_sheet(self):
        """Sheet 2: Cost Sheet (Task-Level)"""
        ws = self.wb.create_sheet("2. Cost Sheet - Tasks")
        
        # Header
        ws.merge_cells('A1:G1')
        ws['A1'] = f"{self.project_title} - Task-Level Costs"
        ws['A1'].font = Font(bold=True, size=14)
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        
        # Column headers
        headers = ["WBS ID", "Task Name", "Project Phase", "Assigned Resource", 
                   "Planned Hours", "Hourly Rate (USD)", "Task Cost (USD)"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Data rows
        row = 4
        total_cost = 0
        for task in self.wbs_tasks:
            ws.cell(row=row, column=1, value=task["wbs_id"])
            ws.cell(row=row, column=2, value=task["task"])
            ws.cell(row=row, column=3, value=task["phase"])
            ws.cell(row=row, column=4, value=task["resource"])
            ws.cell(row=row, column=5, value=task["hours"])
            
            hourly_rate = self.resources[task["resource"]]["hourly_rate"]
            ws.cell(row=row, column=6, value=hourly_rate)
            ws.cell(row=row, column=6).number_format = '$#,##0.00'
            
            task_cost = task["hours"] * hourly_rate
            ws.cell(row=row, column=7, value=task_cost)
            ws.cell(row=row, column=7).number_format = '$#,##0.00'
            total_cost += task_cost
            
            # Apply borders
            for col in range(1, 8):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        # Total row
        ws.cell(row=row, column=1, value="TOTAL")
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=7, value=total_cost)
        ws.cell(row=row, column=7).number_format = '$#,##0.00'
        ws.cell(row=row, column=7).font = Font(bold=True)
        ws.cell(row=row, column=7).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        # Column widths
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 50
        ws.column_dimensions['C'].width = 28
        ws.column_dimensions['D'].width = 22
        ws.column_dimensions['E'].width = 15
        ws.column_dimensions['F'].width = 18
        ws.column_dimensions['G'].width = 18
        
        return total_cost
    
    def create_time_phased_budget(self):
        """Sheet 3: Time-Phased Budget"""
        ws = self.wb.create_sheet("3. Time-Phased Budget")
        
        # Header
        ws.merge_cells('A1:D1')
        ws['A1'] = f"{self.project_title} - Time-Phased Budget (8 Months)"
        ws['A1'].font = Font(bold=True, size=14)
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        
        # Column headers
        headers = ["Month", "Planned Activities", "Resource Cost (USD)", "Cumulative Cost (USD)"]
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col)
            cell.value = header
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            cell.border = Border(
                left=Side(style='thin'),
                right=Side(style='thin'),
                top=Side(style='thin'),
                bottom=Side(style='thin')
            )
        
        # Calculate monthly costs
        monthly_data = {}
        for month in self.months:
            monthly_data[month] = {
                "activities": [],
                "cost": 0
            }
        
        for task in self.wbs_tasks:
            month = task["month"]
            hourly_rate = self.resources[task["resource"]]["hourly_rate"]
            task_cost = task["hours"] * hourly_rate
            
            monthly_data[month]["cost"] += task_cost
            monthly_data[month]["activities"].append(task["phase"])
        
        # Write monthly data
        row = 4
        cumulative_cost = 0
        for month in self.months:
            ws.cell(row=row, column=1, value=month)
            
            # Get unique activities for the month
            activities = list(set(monthly_data[month]["activities"]))
            activities_str = "\n".join([f"• {act}" for act in activities])
            ws.cell(row=row, column=2, value=activities_str)
            ws.cell(row=row, column=2).alignment = Alignment(wrap_text=True, vertical='top')
            
            monthly_cost = monthly_data[month]["cost"]
            ws.cell(row=row, column=3, value=monthly_cost)
            ws.cell(row=row, column=3).number_format = '$#,##0.00'
            
            cumulative_cost += monthly_cost
            ws.cell(row=row, column=4, value=cumulative_cost)
            ws.cell(row=row, column=4).number_format = '$#,##0.00'
            
            # Apply borders
            for col in range(1, 5):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            
            # Set row height for better readability
            ws.row_dimensions[row].height = 60
            row += 1
        
        # Total row
        ws.cell(row=row, column=1, value="TOTAL")
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=3, value=cumulative_cost)
        ws.cell(row=row, column=3).number_format = '$#,##0.00'
        ws.cell(row=row, column=3).font = Font(bold=True)
        ws.cell(row=row, column=3).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        # Column widths
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 50
        ws.column_dimensions['C'].width = 22
        ws.column_dimensions['D'].width = 25
        
        return cumulative_cost
    
    def create_budget_summary(self, direct_cost):
        """Sheet 4: Budget Summary"""
        ws = self.wb.create_sheet("4. Budget Summary")
        
        # Header
        ws.merge_cells('A1:B1')
        ws['A1'] = f"{self.project_title} - Budget Summary"
        ws['A1'].font = Font(bold=True, size=14)
        ws['A1'].alignment = Alignment(horizontal='center')
        ws['A1'].fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
        
        # Project Info
        row = 3
        info = [
            ("Project Title:", self.project_title),
            ("Duration:", f"{self.duration_months} months (October 2025 - May 2026)"),
            ("Methodology:", "Agile-Scrum"),
            ("Domain:", "FinTech / AI / Edge Computing"),
            ("", ""),
        ]
        
        for label, value in info:
            ws.cell(row=row, column=1, value=label).font = Font(bold=True)
            ws.cell(row=row, column=2, value=value)
            row += 1
        
        # Cost Breakdown Header
        row += 1
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "BUDGET BREAKDOWN"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        
        row += 1
        
        # Direct costs
        ws.cell(row=row, column=1, value="Total Direct Resource Cost:")
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=2, value=direct_cost)
        ws.cell(row=row, column=2).number_format = '$#,##0.00'
        ws.cell(row=row, column=2).fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
        row += 1
        
        # Calculation note
        ws.cell(row=row, column=1, value="Calculation:")
        ws.cell(row=row, column=2, value="Sum of (Hours × Hourly Rate) for all tasks")
        ws.cell(row=row, column=2).font = Font(italic=True, size=9)
        row += 2
        
        # Overhead
        overhead_cost = direct_cost * self.overhead_rate
        ws.cell(row=row, column=1, value="Overhead Cost (30%):")
        ws.cell(row=row, column=1).font = Font(bold=True)
        ws.cell(row=row, column=2, value=overhead_cost)
        ws.cell(row=row, column=2).number_format = '$#,##0.00'
        ws.cell(row=row, column=2).fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
        row += 1
        
        # Overhead breakdown
        ws.cell(row=row, column=1, value="Includes:")
        ws.cell(row=row, column=2, value="Infrastructure, Tools, Admin, Contingency")
        ws.cell(row=row, column=2).font = Font(italic=True, size=9)
        row += 2
        
        # Total budget
        total_budget = direct_cost + overhead_cost
        ws.cell(row=row, column=1, value="FINAL TOTAL PROJECT BUDGET:")
        ws.cell(row=row, column=1).font = Font(bold=True, size=12)
        ws.cell(row=row, column=2, value=total_budget)
        ws.cell(row=row, column=2).number_format = '$#,##0.00'
        ws.cell(row=row, column=2).font = Font(bold=True, size=12)
        ws.cell(row=row, column=2).fill = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        
        # Formula note
        row += 1
        ws.cell(row=row, column=1, value="Formula:")
        ws.cell(row=row, column=2, value="= Direct Cost + (Direct Cost × 30%)")
        ws.cell(row=row, column=2).font = Font(italic=True, size=9)
        
        row += 3
        
        # Additional breakdown
        ws.merge_cells(f'A{row}:B{row}')
        ws[f'A{row}'] = "COST DISTRIBUTION BY PHASE"
        ws[f'A{row}'].font = Font(bold=True, size=12)
        ws[f'A{row}'].fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        ws[f'A{row}'].alignment = Alignment(horizontal='center')
        row += 1
        
        # Calculate phase-wise costs
        phase_costs = {}
        for task in self.wbs_tasks:
            phase = task["phase"]
            hourly_rate = self.resources[task["resource"]]["hourly_rate"]
            task_cost = task["hours"] * hourly_rate
            
            if phase not in phase_costs:
                phase_costs[phase] = 0
            phase_costs[phase] += task_cost
        
        # Write phase costs
        for phase, cost in sorted(phase_costs.items()):
            ws.cell(row=row, column=1, value=phase)
            ws.cell(row=row, column=2, value=cost)
            ws.cell(row=row, column=2).number_format = '$#,##0.00'
            
            # Apply borders
            for col in range(1, 3):
                ws.cell(row=row, column=col).border = Border(
                    left=Side(style='thin'),
                    right=Side(style='thin'),
                    top=Side(style='thin'),
                    bottom=Side(style='thin')
                )
            row += 1
        
        # Column widths
        ws.column_dimensions['A'].width = 35
        ws.column_dimensions['B'].width = 35
        
        return total_budget
    
    def generate_budget(self, output_file="Project_Budget_Federated_Learning.xlsx"):
        """Main method to generate the complete budget"""
        print("=" * 70)
        print("TIME-PHASED PROJECT BUDGET GENERATOR")
        print("=" * 70)
        print(f"Project: {self.project_title}")
        print(f"Duration: {self.duration_months} months (October 2025 - May 2026)")
        print(f"Methodology: Agile-Scrum")
        print("=" * 70)
        
        # Calculate totals
        print("\n[1/5] Calculating resource totals...")
        self.calculate_resource_totals()
        
        # Create sheets
        print("[2/5] Creating Resource Sheet...")
        direct_cost = self.create_resource_sheet()
        
        print("[3/5] Creating Cost Sheet (Task-Level)...")
        task_total = self.create_cost_sheet()
        
        print("[4/5] Creating Time-Phased Budget...")
        time_phased_total = self.create_time_phased_budget()
        
        print("[5/5] Creating Budget Summary...")
        total_budget = self.create_budget_summary(direct_cost)
        
        # Save workbook
        output_path = os.path.join(os.getcwd(), output_file)
        self.wb.save(output_path)
        
        print("\n" + "=" * 70)
        print("BUDGET GENERATION COMPLETE!")
        print("=" * 70)
        print(f"\n📊 BUDGET SUMMARY:")
        print(f"   Direct Resource Cost:  ${direct_cost:,.2f}")
        print(f"   Overhead (30%):        ${direct_cost * 0.30:,.2f}")
        print(f"   TOTAL PROJECT BUDGET:  ${total_budget:,.2f}")
        print(f"\n📁 File saved: {output_path}")
        print(f"📝 Total Tasks: {len(self.wbs_tasks)}")
        print(f"👥 Total Resources: {len(self.resources)}")
        print("=" * 70)
        
        return output_path

if __name__ == "__main__":
    generator = ProjectBudgetGenerator()
    generator.generate_budget()
