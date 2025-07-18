from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Get a value from a dictionary by key."""
    if dictionary is None:
        return None
    return dictionary.get(key) 