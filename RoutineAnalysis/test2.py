import altair as alt
from vega_datasets import data

alt.renderers.enable("html")

source = data.population.url

chart = (
    alt.Chart(source)
    .mark_area()
    .encode(
        x="age:O",
        y=alt.Y("sum(people):Q", title="Population"),
        facet=alt.Facet("year:O", columns=5),
    )
    .properties(width=100, height=80)
)

chart.save("filename.html")
