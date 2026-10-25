/* Version dropdown for the sidebar.
 *
 * Locates the site root from the current URL (pages live under
 * /<major.minor>/, possibly below a subpath) and fetches versions.json from
 * there at runtime, so even old, frozen snapshots list all versions published
 * since. Fails silently when the file is unreachable (e.g. a local docs
 * build opened directly), leaving the sidebar untouched.
 */
(function () {
  function init() {
    var match = window.location.pathname.match(/^(.*?)\/(\d+\.\d+)\//);
    var root = match ? match[1] : "";
    var current = match ? match[2] : null;

    fetch(root + "/versions.json")
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        if (!data || !data.versions || !data.versions.length) return;
        var box = document.querySelector(".wy-side-nav-search");
        if (!box) return;

        var select = document.createElement("select");
        select.className = "version-switcher";
        select.setAttribute("aria-label", "Documentation version");
        data.versions.forEach(function (v) {
          var opt = document.createElement("option");
          opt.value = root + v.url;
          opt.textContent = "v" + v.version + (v.version === data.latest ? " (latest)" : "");
          if (v.version === current) opt.selected = true;
          select.appendChild(opt);
        });
        select.addEventListener("change", function () {
          if (select.value) window.location.href = select.value;
        });
        box.appendChild(select);
      })
      .catch(function () {});
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
