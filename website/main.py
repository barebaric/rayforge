def on_pre_page_macros(env):
    """
    This hook from the macros plugin is called for each page before it is
    rendered. We use it to manually construct and overwrite the
    `page.edit_url`.
    """
    page = env.page
    config = env.conf

    # Check if repo_url is configured. If not, we can't build the URL.
    if not config.get("repo_url"):
        return

    repo_url = config["repo_url"]
    # Ensure the repo_url doesn't have a trailing slash
    if repo_url.endswith("/"):
        repo_url = repo_url[:-1]

    # Check which environment we are running on
    extra = config["extra"]
    is_prod = extra["env"] == "production"

    # On prod we have to translate the path to match what is on Github.
    # (e.g., 'docs/latest/foo' or 'docs/0.1.2/foo' becomes 'docs/foo')
    path_parts = page.file.src_path.split("/")

    if is_prod and path_parts[0] == "docs":
        try:
            path = "docs/" + "/".join(path_parts[2:])
        except ValueError:
            return
    else:
        path = page.file.src_path

    # Overwrite the page's edit_url with our correct one.
    page.edit_url = f"{repo_url}/edit/main/website/content/{path}"
