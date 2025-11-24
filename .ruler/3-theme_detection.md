# Trial 5: Parent CSS Variables
# Checks if we can read CSS variables from the parent document.
js = """
const style = getComputedStyle(window.parent.document.documentElement);
const bg = style.getPropertyValue('--vscode-editor-background');
document.getElementById('result').innerText = 'VS Code Bg Var (Parent): ' + (bg || 'Not found');
"""
create_test_widget(js, "Trial 5: Parent CSS Variables")

this works for bg color!

# Trial 6: Body Attribute
# Checks for data-vscode-theme-kind attribute on parent body.
js = """
const kind = window.parent.document.body.getAttribute('data-vscode-theme-kind');
document.getElementById('result').innerText = 'Theme Kind Attr: ' + (kind || 'Not found');
"""
create_test_widget(js, "Trial 6: Body Attribute")

this works for light/dark theme. you just need to search for "dark" or "light" in the text

