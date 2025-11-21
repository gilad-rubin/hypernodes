import graphviz

def create_modern_graph():
    dot = graphviz.Digraph('modern_ui', comment='Modern UI Style Graph')
    
    # Graph Attributes
    dot.attr(
        rankdir='TB',
        splines='ortho',
        nodesep='0.8',
        ranksep='1.0',
        bgcolor='transparent',
        fontname='Helvetica',
        pad='0.5'
    )
    
    # Default Node Attributes
    dot.attr('node', 
        shape='plain', # We use HTML labels, so shape is plain
        fontname='Helvetica',
        fontsize='12'
    )
    
    # Default Edge Attributes
    dot.attr('edge',
        color='#D1D5DB', # Light gray
        penwidth='1.5',
        arrowhead='none', # We'll customize arrowheads
        arrowsize='0.8'
    )

    # Helper to create an HTML label
    def make_label(title, subtitle, icon_color, icon_char, badge_text=None, badge_color=None):
        # CSS-like inline styles are limited in Graphviz, so we use table attributes
        
        badge_html = ""
        if badge_text:
            badge_html = f'''
            <TD ALIGN="RIGHT" VALIGN="TOP">
                <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                    <TR><TD BGCOLOR="{badge_color}" STYLE="ROUNDED" WIDTH="20" HEIGHT="20"><FONT COLOR="WHITE" POINT-SIZE="10">{badge_text}</FONT></TD></TR>
                </TABLE>
            </TD>
            '''
        else:
            badge_html = '<TD></TD>'

        label = f'''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0" WIDTH="250" HEIGHT="100" BGCOLOR="WHITE" STYLE="ROUNDED">
            <!-- Top Padding -->
            <TR><TD HEIGHT="15" COLSPAN="3"></TD></TR>
            
            <TR>
                <!-- Icon Column -->
                <TD WIDTH="20"></TD>
                <TD WIDTH="40" VALIGN="TOP">
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" BGCOLOR="{icon_color}" WIDTH="40" HEIGHT="40">
                        <TR><TD ALIGN="CENTER" VALIGN="MIDDLE"><FONT COLOR="WHITE" POINT-SIZE="18">{icon_char}</FONT></TD></TR>
                    </TABLE>
                </TD>
                <TD WIDTH="15"></TD>
                
                <!-- Content Column -->
                <TD VALIGN="TOP" ALIGN="LEFT">
                    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                        <TR><TD ALIGN="LEFT"><B><FONT POINT-SIZE="14" COLOR="#111827">{title}</FONT></B></TD></TR>
                        <TR><TD HEIGHT="4"></TD></TR>
                        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11" COLOR="#6B7280">{subtitle}</FONT></TD></TR>
                    </TABLE>
                </TD>
                
                <!-- Right Padding/Badge -->
                <TD WIDTH="10"></TD>
                {badge_html}
                <TD WIDTH="15"></TD>
            </TR>
            
            <!-- Bottom Padding -->
            <TR><TD HEIGHT="15" COLSPAN="3"></TD></TR>
            
            <!-- Footer Row (e.g. Automatic/Time) -->
             <TR>
                <TD COLSPAN="2"></TD>
                <TD COLSPAN="3" ALIGN="LEFT">
                     <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#9CA3AF">Automatic</FONT></TD></TR>
                    </TABLE>
                </TD>
                <TD COLSPAN="2"></TD>
            </TR>
             <TR><TD HEIGHT="15" COLSPAN="3"></TD></TR>
        </TABLE>
        >'''
        return label

    # Define Nodes
    
    # Node 1: Client Onboarding
    label1 = make_label("Client onboarding", "The customer is assigned...", "#3B82F6", "C", "S", "#EF4444")
    dot.node('node1', label=label1)

    # Node 2: Trivial
    label2 = make_label("Trivial", "The customer is discarded...", "#EF4444", "X")
    dot.node('node2', label=label2)

    # Node 3: Onboarding Email
    label3 = make_label("3.1. Onboarding", "A sales lead messages...", "#F59E0B", "M")
    dot.node('node3', label=label3)

    # Node 4: Split Decision (Diamond)
    # Graphviz HTML labels don't support diamond shape well with complex content inside.
    # We often simulate this or just use a standard diamond node.
    # For the "Split decision" look, we might use a small circle or diamond node.
    dot.node('split', label='', shape='diamond', style='filled', fillcolor='#F3F4F6', color='#D1D5DB', width='0.5', height='0.5')
    
    # Node 5: Response
    dot.node('response', label='Response', shape='rect', style='rounded,filled', fillcolor='#F3F4F6', color='#D1D5DB', fontcolor='#374151', width='1.5')

    # Node 6: Invoice Prep
    label6 = make_label("4.1. Invoice prep", "Data sent to Hubspot...", "#F97316", "I")
    dot.node('node6', label=label6)

    # Node 7: Intro Meeting
    label7 = make_label("4.2. Intro meeting", "Sales lead gets more info...", "#3B82F6", "M", "S", "#3B82F6")
    dot.node('node7', label=label7)


    # Define Edges
    dot.edge('node1', 'node3', color='#8B5CF6', penwidth='2.0') # Purple edge
    dot.edge('node2', 'node3', style='invis') # Invisible edge for layout if needed
    
    dot.edge('node3', 'response', color='#8B5CF6', penwidth='2.0')
    dot.edge('response', 'split', color='#8B5CF6', penwidth='2.0')
    
    dot.edge('split', 'node6', label='agree', fontsize='10', fontcolor='#6B7280', color='#D1D5DB')
    dot.edge('split', 'node7', label='more info', fontsize='10', fontcolor='#6B7280', color='#D1D5DB')

    # Render
    output_path = dot.render('modern_graph', format='png', cleanup=True)
    print(f"Graph generated at: {output_path}")

if __name__ == "__main__":
    create_modern_graph()
