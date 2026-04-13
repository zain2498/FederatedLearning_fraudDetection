"""
Critical Path Diagram Generator for Federated Learning Project
Using CPM (Critical Path Method) principles
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_critical_path_diagram():
    """
    Generate a professional Critical Path Diagram for the 
    Federated Learning Edge Computing Fraud Detection Project
    """
    
    # Create figure with high DPI for professional quality
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define critical path activities
    activities = [
        {'id': '1.1.10', 'name': 'Project Charter\nApproval', 'pos': (1, 4.5)},
        {'id': '1.2.13', 'name': 'Planning Phase\nApproval', 'pos': (3, 4.5)},
        {'id': '2.5.4', 'name': 'Requirements\nSign-off', 'pos': (5, 4.5)},
        {'id': '3.5.3', 'name': 'Design\nApproval', 'pos': (7, 4.5)},
        {'id': '4.4.4', 'name': 'Validate System\nIntegration', 'pos': (7, 2)},
        {'id': '5.5.3', 'name': 'Testing\nSign-off', 'pos': (5, 2)},
        {'id': '6.3.1', 'name': 'Deployment\nApproval', 'pos': (3, 2)},
        {'id': '7.3.3', 'name': 'Final Project\nSubmission', 'pos': (1, 2)},
    ]
    
    # Define dependencies (from -> to)
    dependencies = [
        (0, 1),  # Charter -> Planning
        (1, 2),  # Planning -> Requirements
        (2, 3),  # Requirements -> Design
        (3, 4),  # Design -> Integration
        (4, 5),  # Integration -> Testing
        (5, 6),  # Testing -> Deployment
        (6, 7),  # Deployment -> Submission
    ]
    
    # Draw activity boxes (critical path - red color)
    boxes = []
    for activity in activities:
        x, y = activity['pos']
        
        # Create fancy box with shadow effect
        box = FancyBboxPatch(
            (x - 0.5, y - 0.3),
            1.0, 0.6,
            boxstyle="round,pad=0.05",
            edgecolor='#C41E3A',  # Critical path red
            facecolor='#FFE6E6',  # Light red fill
            linewidth=3,
            zorder=2
        )
        ax.add_patch(box)
        boxes.append((x, y))
        
        # Add task ID (bold)
        ax.text(x, y + 0.15, activity['id'],
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                fontfamily='Arial',
                zorder=3)
        
        # Add task name
        ax.text(x, y - 0.05, activity['name'],
                ha='center', va='center',
                fontsize=9,
                fontfamily='Arial',
                zorder=3)
    
    # Draw dependency arrows
    for from_idx, to_idx in dependencies:
        x1, y1 = activities[from_idx]['pos']
        x2, y2 = activities[to_idx]['pos']
        
        # Adjust start and end points to box edges
        if x1 < x2:  # Horizontal arrow (left to right)
            start_x, start_y = x1 + 0.5, y1
            end_x, end_y = x2 - 0.5, y2
        elif x1 > x2:  # Horizontal arrow (right to left)
            start_x, start_y = x1 - 0.5, y1
            end_x, end_y = x2 + 0.5, y2
        else:  # Vertical arrow
            start_x, start_y = x1, y1 - 0.3
            end_x, end_y = x2, y2 + 0.3
        
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='#C41E3A',
            linewidth=2.5,
            zorder=1
        )
        ax.add_patch(arrow)
    
    # Add START and END markers
    ax.text(0.3, 4.5, 'START',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#90EE90', edgecolor='#228B22', linewidth=2),
            fontfamily='Arial')
    
    ax.text(9.7, 2, 'END',
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#90EE90', edgecolor='#228B22', linewidth=2),
            fontfamily='Arial')
    
    # Draw connecting arrows to START and END
    start_arrow = FancyArrowPatch(
        (0.6, 4.5), (0.5, 4.5),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='#228B22', linewidth=2.5, zorder=1
    )
    ax.add_patch(start_arrow)
    
    end_arrow = FancyArrowPatch(
        (1.5, 2), (9.4, 2),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='#228B22', linewidth=2.5, zorder=1
    )
    ax.add_patch(end_arrow)
    
    # Add title
    ax.text(5, 5.6, 'Critical Path Diagram',
            ha='center', va='center',
            fontsize=18, fontweight='bold',
            fontfamily='Arial')
    
    ax.text(5, 5.3, 'Federated Learning for Edge Computing - Fraud Detection Project',
            ha='center', va='center',
            fontsize=12,
            fontfamily='Arial',
            style='italic')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='#FFE6E6', edgecolor='#C41E3A', linewidth=2, label='Critical Path Activity (TF = 0)'),
        mpatches.FancyArrow(0, 0, 0.3, 0, width=0.1, color='#C41E3A', label='FS Dependency'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, fancybox=True)
    
    # Add project information box
    info_text = (
        "Project Duration: ~8 months\n"
        "Methodology: Agile-Scrum\n"
        "All dependencies: Finish-to-Start (FS)\n"
        "Total Float: 0 (All activities critical)"
    )
    ax.text(0.5, 0.5, info_text,
            ha='left', va='bottom',
            fontsize=9,
            fontfamily='Arial',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='#808080', linewidth=1))
    
    # Add CPM note
    ax.text(9.5, 5.3, 'CPM Analysis',
            ha='right', va='center',
            fontsize=10, fontweight='bold',
            fontfamily='Arial',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFFACD', edgecolor='#FFD700', linewidth=2))
    
    plt.tight_layout()
    
    return fig

def main():
    """Generate and save the critical path diagram"""
    print("🎨 Generating Critical Path Diagram...")
    
    fig = create_critical_path_diagram()
    
    # Save as high-quality PNG
    output_file = 'critical_path_diagram.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Diagram saved as: {output_file}")
    
    # Save as PDF for vector graphics (best for reports)
    pdf_file = 'critical_path_diagram.pdf'
    fig.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✅ Diagram saved as: {pdf_file}")
    
    # Save as SVG for editing in Inkscape/Illustrator
    svg_file = 'critical_path_diagram.svg'
    fig.savefig(svg_file, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✅ Diagram saved as: {svg_file}")
    
    print("\n📊 Critical Path Sequence:")
    print("START → 1.1.10 → 1.2.13 → 2.5.4 → 3.5.3 → 4.4.4 → 5.5.3 → 6.3.1 → 7.3.3 → END")
    print("\n💡 You can now insert these files into:")
    print("   - PowerPoint presentations")
    print("   - Word documents")
    print("   - Academic reports")
    print("   - Project documentation")

if __name__ == "__main__":
    main()
