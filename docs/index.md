<ul>
  {% for page in site.pages %}
    <li><a href="{{ page.url | relative_url }}">{{ page.title | default: page.name }}</a></li>
  {% endfor %}
</ul>

