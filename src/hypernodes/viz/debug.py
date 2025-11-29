"""
Debugging utilities for visualization VSCode compatibility issues.

Usage:
    from hypernodes.viz.debug import diagnose_widget, quick_check
    
    # Quick check
    quick_check(pipeline)
    
    # Full diagnosis
    result = diagnose_widget(pipeline)
    print(result["report"])
"""

import html as html_module
import re
from typing import Any, Dict

from .visualization_widget import PipelineWidget


def quick_check(pipeline: Any, verbose: bool = True) -> bool:
    """
    Quick compatibility check for a pipeline visualization.
    
    Args:
        pipeline: Pipeline to check
        verbose: Print results to console
        
    Returns:
        True if all checks pass, False otherwise
    """
    result = diagnose_widget(pipeline)
    
    if verbose:
        print(result["report"])
    
    return result["compatible"]


def diagnose_widget(pipeline: Any) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of widget VSCode compatibility.
    
    Args:
        pipeline: Pipeline to diagnose
        
    Returns:
        Dict with:
        - compatible: bool - True if all checks pass
        - checks: dict - Individual check results
        - issues: list - List of issue descriptions
        - report: str - Human-readable report
        - html_size: int - Size of generated HTML in bytes
    """
    widget = PipelineWidget(pipeline)
    decoded = html_module.unescape(widget.value)
    
    # Run all checks
    checks = {}
    issues = []
    
    # Check 1: No module scripts
    module_matches = re.findall(
        r'<script[^>]*type\s*=\s*["\']module["\'][^>]*>',
        decoded,
        re.IGNORECASE
    )
    checks["no_module_scripts"] = len(module_matches) == 0
    if module_matches:
        issues.append(
            f"Found {len(module_matches)} module script(s) - "
            "these don't execute in VSCode iframe srcdoc"
        )
    
    # Check 2: Has DOMContentLoaded
    checks["has_dom_content_loaded"] = "DOMContentLoaded" in decoded
    if not checks["has_dom_content_loaded"]:
        issues.append(
            "Missing DOMContentLoaded wrapper - "
            "scripts may run before DOM is ready"
        )
    
    # Check 3: Has IIFE
    checks["has_iife"] = "(function()" in decoded
    if not checks["has_iife"]:
        issues.append("Missing IIFE pattern for scope isolation")
    
    # Check 4: Has strict mode
    checks["has_strict_mode"] = "'use strict'" in decoded
    if not checks["has_strict_mode"]:
        issues.append("Missing 'use strict' in IIFE")
    
    # Check 5: Uses srcdoc
    checks["uses_srcdoc"] = "srcdoc=" in widget.value
    if not checks["uses_srcdoc"]:
        issues.append(
            "Not using srcdoc attribute - "
            "base64 data URIs may be blocked"
        )
    
    # Check 6: Has graph-data element
    checks["has_graph_data"] = 'id="graph-data"' in decoded
    if not checks["has_graph_data"]:
        issues.append("Missing graph-data element")
    
    # Check 7: Graph data is valid JSON
    json_match = re.search(
        r'<script[^>]*id="graph-data"[^>]*>(.*?)</script>',
        decoded,
        re.DOTALL
    )
    if json_match:
        try:
            import json
            json.loads(json_match.group(1))
            checks["valid_graph_json"] = True
        except (json.JSONDecodeError, Exception) as e:
            checks["valid_graph_json"] = False
            issues.append(f"Invalid JSON in graph-data: {e}")
    else:
        checks["valid_graph_json"] = False
        issues.append("Could not find graph-data content")
    
    # Build report
    compatible = len(issues) == 0
    
    report_lines = [
        "=" * 50,
        "VSCode Visualization Compatibility Report",
        "=" * 50,
        "",
    ]
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        readable_name = check_name.replace("_", " ").title()
        report_lines.append(f"{status} {readable_name}")
    
    report_lines.append("")
    report_lines.append(f"HTML size: {len(widget.value):,} bytes")
    report_lines.append("")
    
    if compatible:
        report_lines.append("✅ All checks passed! Widget should work in VSCode.")
    else:
        report_lines.append("❌ Issues found:")
        for issue in issues:
            report_lines.append(f"   • {issue}")
        report_lines.append("")
        report_lines.append("See: skills/viz/js/VSCODE_NOTEBOOK_COMPATIBILITY.md")
    
    report_lines.append("=" * 50)
    
    return {
        "compatible": compatible,
        "checks": checks,
        "issues": issues,
        "report": "\n".join(report_lines),
        "html_size": len(widget.value),
    }


def extract_js_errors(widget_html: str) -> list:
    """
    Extract potential JS error sources from widget HTML.
    
    Looks for common patterns that cause errors in VSCode notebooks.
    
    Args:
        widget_html: Raw widget HTML value
        
    Returns:
        List of potential error descriptions
    """
    decoded = html_module.unescape(widget_html)
    errors = []
    
    # Check for getElementById before element
    # Look for patterns where getElementById is called in inline script
    # before the referenced element
    id_refs = re.findall(r'getElementById\(["\']([^"\']+)["\']\)', decoded)
    
    for ref_id in set(id_refs):
        # Find first reference and element position
        ref_pos = decoded.find(f'getElementById("{ref_id}")')
        if ref_pos == -1:
            ref_pos = decoded.find(f"getElementById('{ref_id}')")
        
        elem_pos = decoded.find(f'id="{ref_id}"')
        if elem_pos == -1:
            elem_pos = decoded.find(f"id='{ref_id}'")
        
        if elem_pos == -1:
            errors.append(f"Element #{ref_id} referenced but not found in HTML")
        elif ref_pos < elem_pos and "DOMContentLoaded" not in decoded[:ref_pos]:
            errors.append(
                f"Element #{ref_id} referenced at pos {ref_pos} "
                f"but defined at pos {elem_pos} - needs DOMContentLoaded"
            )
    
    return errors


# For convenience, expose at module level
__all__ = ["quick_check", "diagnose_widget", "extract_js_errors"]
