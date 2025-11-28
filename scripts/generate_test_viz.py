"""Generate test visualization HTML."""
from hypernodes import Pipeline, node
from hypernodes.viz.ui_handler import UIHandler
from hypernodes.viz.js.renderer import JSRenderer
from hypernodes.viz.js.html_generator import generate_widget_html


@node(output_name='sum')
def add(x: int, y: int) -> int:
    return x + y


@node(output_name=('product', 'remainder'))
def multiply(sum: int, y: int) -> tuple[float, int]:
    return sum * y, sum % y


@node(output_name='result')
def combine(product: float, sum: int) -> dict:
    return {'product': product, 'sum': sum}


def main():
    pipeline = Pipeline(nodes=[add, multiply, combine])

    # Generate the HTML
    handler = UIHandler(pipeline, depth=1)
    graph_data = handler.get_visualization_data(traverse_collapsed=True)

    renderer = JSRenderer()
    react_flow_data = renderer.render(
        graph_data,
        theme='dark',
        separate_outputs=False,
        show_types=True,
    )

    html_content = generate_widget_html(react_flow_data)

    # Write to file
    with open('outputs/test_integrated_toggles.html', 'w') as f:
        f.write(html_content)

    print('HTML saved to outputs/test_integrated_toggles.html')


if __name__ == "__main__":
    main()

