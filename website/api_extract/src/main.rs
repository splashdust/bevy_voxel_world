//! Builds the crate's rustdoc JSON with a nightly toolchain and
//! writes `website/api.json`, the data behind the site's docs section: each
//! public item with its signature, docs, members and the cargo features it
//! needs.
//!
//! Features are found by building once with every feature in `API_FEATURES`
//! and once without each of them. An item, member or trait impl that is missing
//! from the build without a feature needs that feature.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path as FsPath, PathBuf};
use std::process::Command;

use rustdoc_types::{
    Crate, FORMAT_VERSION, Function, GenericArg, GenericArgs, GenericBound, GenericParamDef,
    GenericParamDefKind, Generics, Id, Item, ItemEnum, ItemKind, ItemSummary, Path, StructKind,
    Term, TraitBoundModifier, Type, VariantKind, Visibility, WherePredicate,
};
use serde::{Deserialize, Serialize};

const CRATE: &str = "bevy_voxel_world";

/// The features that add public API. `trace` and `trace_tracy` add none, and
/// building `trace_tracy` compiles Tracy's C++ client.
const API_FEATURES: &[&str] = &["light_propagation", "sub_grids", "dislodged_fragments", "serde"];

/// Trait impls left off the site: the plumbing that `#[derive(Reflect)]`,
/// `#[derive(Asset)]` and `#[derive(PartialEq)]` add beside the trait a reader
/// cares about.
const HIDDEN_TRAITS: &[&str] = &[
    "FromReflect",
    "GetTypeRegistration",
    "PartialReflect",
    "Struct",
    "Enum",
    "TupleStruct",
    "TypePath",
    "Typed",
    "StructuralPartialEq",
    "VisitAssetDependencies",
];

fn main() {
    let website = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let repo = website.parent().unwrap().to_path_buf();
    let toolchain = std::env::var("API_DOCS_TOOLCHAIN").unwrap_or_else(|_| "nightly".into());
    let metadata = cargo_metadata(&repo);

    let all = build(&repo, &toolchain, &metadata.target_directory, API_FEATURES);
    let without: Vec<(&str, HashSet<String>)> = API_FEATURES
        .iter()
        .map(|&feature| {
            let rest: Vec<&str> = API_FEATURES.iter().copied().filter(|&f| f != feature).collect();
            let krate = build(&repo, &toolchain, &metadata.target_directory, &rest);
            (feature, keys(&krate))
        })
        .collect();

    let api = Extractor::new(&all, &metadata.versions, &without).run();
    let out = website.join("api.json");
    serde_json::to_writer(File::create(&out).unwrap(), &api).unwrap();
    eprintln!("wrote {} items to {}", api.items.len(), out.display());
}

struct Metadata {
    target_directory: PathBuf,
    /// Each dependency's version, by its library name.
    versions: HashMap<String, String>,
}

fn cargo_metadata(repo: &FsPath) -> Metadata {
    #[derive(Deserialize)]
    struct Raw {
        target_directory: PathBuf,
        packages: Vec<Package>,
    }
    #[derive(Deserialize)]
    struct Package {
        name: String,
        version: String,
    }
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--all-features", "--manifest-path"])
        .arg(repo.join("Cargo.toml"))
        .output()
        .expect("cargo runs");
    assert!(output.status.success(), "cargo metadata failed");
    let raw: Raw = serde_json::from_slice(&output.stdout).unwrap();
    Metadata {
        target_directory: raw.target_directory,
        versions: raw
            .packages
            .into_iter()
            .map(|p| (p.name.replace('-', "_"), p.version))
            .collect(),
    }
}

/// Builds the crate's rustdoc JSON with the given features and reads it.
fn build(repo: &FsPath, toolchain: &str, target_dir: &FsPath, features: &[&str]) -> Crate {
    eprintln!("> rustdoc JSON with features [{}]", features.join(", "));
    let status = Command::new("cargo")
        .env("RUSTUP_TOOLCHAIN", toolchain)
        .args(["rustdoc", "--lib", "--no-default-features", "--manifest-path"])
        .arg(repo.join("Cargo.toml"))
        .arg("--features")
        .arg(features.join(","))
        .args(["--", "-Z", "unstable-options", "--output-format", "json"])
        .status()
        .expect("cargo runs");
    assert!(status.success(), "rustdoc failed");

    let file = target_dir.join("doc").join(format!("{CRATE}.json"));
    #[derive(Deserialize)]
    struct Version {
        format_version: u32,
    }
    let version: Version = serde_json::from_reader(BufReader::new(File::open(&file).unwrap())).unwrap();
    assert_eq!(
        version.format_version, FORMAT_VERSION,
        "{toolchain} writes rustdoc JSON format {}, and this tool reads {FORMAT_VERSION}. \
         Set API_DOCS_TOOLCHAIN to a nightly that writes {FORMAT_VERSION}, or update rustdoc-types.",
        version.format_version
    );
    serde_json::from_reader(BufReader::new(File::open(&file).unwrap())).unwrap()
}

/// The public items of a crate, each with every path it can be named by.
struct Public {
    items: Vec<(Id, Vec<String>)>,
    modules: Vec<(Id, String)>,
}

impl Public {
    fn of(krate: &Crate) -> Self {
        let mut public = Public { items: Vec::new(), modules: Vec::new() };
        let mut slots: HashMap<Id, usize> = HashMap::new();
        let root = &krate.index[&krate.root];
        if let ItemEnum::Module(module) = &root.inner {
            for child in &module.items {
                public.visit(krate, *child, None, "", &mut slots);
            }
        }
        public
    }

    fn visit(
        &mut self,
        krate: &Crate,
        id: Id,
        rename: Option<&str>,
        prefix: &str,
        slots: &mut HashMap<Id, usize>,
    ) {
        let Some(item) = krate.index.get(&id) else { return };
        let join = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}::{name}")
            }
        };
        match &item.inner {
            ItemEnum::Use(import) => {
                let Some(target) = import.id else { return };
                if import.is_glob {
                    if let Some(ItemEnum::Module(module)) = krate.index.get(&target).map(|t| &t.inner) {
                        for child in &module.items {
                            self.visit(krate, *child, None, prefix, slots);
                        }
                    }
                } else {
                    self.visit(krate, target, Some(&import.name), prefix, slots);
                }
            }
            ItemEnum::Module(module) => {
                let path = join(rename.or(item.name.as_deref()).unwrap_or_default());
                self.modules.push((id, path.clone()));
                for child in &module.items {
                    self.visit(krate, *child, None, &path, slots);
                }
            }
            _ => {
                let path = join(rename.or(item.name.as_deref()).unwrap_or_default());
                match slots.get(&id) {
                    Some(&slot) => self.items[slot].1.push(path),
                    None => {
                        slots.insert(id, self.items.len());
                        self.items.push((id, vec![path]));
                    }
                }
            }
        }
    }
}

/// The path a reader should import an item by: its prelude path if it has one.
fn canonical(paths: &[String]) -> &str {
    paths.iter().find(|p| p.starts_with("prelude::")).unwrap_or(&paths[0])
}

struct Member<'a> {
    key: String,
    section: &'static str,
    item: &'a Item,
}

/// An item's fields, variants, inherent methods and trait items, each keyed by
/// its kind and name, like `fn.light_at`.
fn members<'a>(krate: &'a Crate, item: &'a Item) -> Vec<Member<'a>> {
    let mut out = Vec::new();
    let mut push = |key: String, section, id: &Id| {
        if let Some(item) = krate.index.get(id) {
            out.push(Member { key, section, item });
        }
    };
    let assoc = |item: &Item, in_trait: bool| -> (String, &'static str) {
        let name = item.name.clone().unwrap_or_default();
        match &item.inner {
            ItemEnum::Function(f) if in_trait && f.has_body => (format!("fn.{name}"), "Provided methods"),
            ItemEnum::Function(_) if in_trait => (format!("fn.{name}"), "Required methods"),
            ItemEnum::Function(_) => (format!("fn.{name}"), "Methods"),
            ItemEnum::AssocType { .. } => (format!("type.{name}"), "Associated types"),
            _ => (format!("const.{name}"), "Associated constants"),
        }
    };
    let impls: &[Id] = match &item.inner {
        ItemEnum::Struct(s) => {
            match &s.kind {
                StructKind::Plain { fields, .. } => {
                    for field in fields {
                        let name = krate.index[field].name.clone().unwrap_or_default();
                        push(format!("field.{name}"), "Fields", field);
                    }
                }
                StructKind::Tuple(fields) => {
                    for (i, field) in fields.iter().enumerate() {
                        if let Some(field) = field {
                            push(format!("field.{i}"), "Fields", field);
                        }
                    }
                }
                StructKind::Unit => {}
            }
            &s.impls
        }
        ItemEnum::Enum(e) => {
            for variant in &e.variants {
                let name = krate.index[variant].name.clone().unwrap_or_default();
                push(format!("variant.{name}"), "Variants", variant);
            }
            &e.impls
        }
        ItemEnum::Trait(t) => {
            for id in &t.items {
                if let Some(member) = krate.index.get(id) {
                    let (key, section) = assoc(member, true);
                    push(key, section, id);
                }
            }
            &[]
        }
        _ => &[],
    };
    for impl_id in impls {
        let ItemEnum::Impl(imp) = &krate.index[impl_id].inner else { continue };
        if imp.trait_.is_some() || imp.is_synthetic || imp.blanket_impl.is_some() {
            continue;
        }
        for id in &imp.items {
            if let Some(member) = krate.index.get(id) {
                if member.visibility == Visibility::Public {
                    let (key, section) = assoc(member, false);
                    push(key, section, id);
                }
            }
        }
    }
    out
}

/// The traits an item implements itself, leaving out blanket and auto-trait
/// impls, as each trait's name and id.
fn trait_impls<'a>(krate: &'a Crate, item: &Item) -> Vec<&'a Path> {
    let impls: &[Id] = match &item.inner {
        ItemEnum::Struct(s) => &s.impls,
        ItemEnum::Enum(e) => &e.impls,
        ItemEnum::Union(u) => &u.impls,
        _ => &[],
    };
    impls
        .iter()
        .filter_map(|id| match &krate.index[id].inner {
            ItemEnum::Impl(imp) if !imp.is_synthetic && imp.blanket_impl.is_none() && !imp.is_negative => {
                imp.trait_.as_ref()
            }
            _ => None,
        })
        .collect()
}

fn last_segment(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// Every path a build can name: each item's paths, and each path followed by
/// a member key or `impl.Trait`.
fn keys(krate: &Crate) -> HashSet<String> {
    let mut keys = HashSet::new();
    for (id, paths) in &Public::of(krate).items {
        let item = &krate.index[id];
        let suffixes: Vec<String> = members(krate, item)
            .into_iter()
            .map(|m| m.key)
            .chain(trait_impls(krate, item).into_iter().map(|t| format!("impl.{}", last_segment(&t.path))))
            .collect();
        for path in paths {
            keys.insert(path.clone());
            for suffix in &suffixes {
                keys.insert(format!("{path}::{suffix}"));
            }
        }
    }
    keys
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Api {
    #[serde(rename = "crate")]
    krate: &'static str,
    version: String,
    format_version: u32,
    features: Vec<&'static str>,
    modules: Vec<ModuleDoc>,
    items: Vec<ItemDoc>,
}

#[derive(Serialize)]
struct ModuleDoc {
    path: String,
    docs: String,
    links: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct ItemDoc {
    path: String,
    aliases: Vec<String>,
    name: String,
    kind: &'static str,
    features: Vec<&'static str>,
    signature: Vec<Token>,
    docs: String,
    links: BTreeMap<String, String>,
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deprecated: Option<String>,
    members: Vec<MemberDoc>,
    traits: Vec<TraitDoc>,
}

#[derive(Serialize)]
struct MemberDoc {
    key: String,
    name: String,
    section: &'static str,
    features: Vec<&'static str>,
    signature: Vec<Token>,
    docs: String,
    links: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct TraitDoc {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    href: Option<String>,
    features: Vec<&'static str>,
}

/// One piece of a signature: text, a link (`h`) or a keyword or name (`k`).
#[derive(Serialize)]
struct Token {
    t: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    h: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    k: Option<&'static str>,
}

struct Extractor<'a> {
    krate: &'a Crate,
    public: Public,
    versions: &'a HashMap<String, String>,
    without: &'a [(&'static str, HashSet<String>)],
    /// Where a link to each public item or member goes: `api:` and its key.
    targets: HashMap<Id, String>,
}

impl<'a> Extractor<'a> {
    fn new(
        krate: &'a Crate,
        versions: &'a HashMap<String, String>,
        without: &'a [(&'static str, HashSet<String>)],
    ) -> Self {
        let public = Public::of(krate);
        let mut targets = HashMap::new();
        for (id, paths) in &public.items {
            let path = canonical(paths);
            targets.insert(*id, format!("api:{path}"));
            for member in members(krate, &krate.index[id]) {
                targets.insert(member.item.id, format!("api:{path}::{}", member.key));
            }
        }
        for (id, path) in &public.modules {
            targets.insert(*id, format!("api:{path}"));
        }
        Extractor { krate, public, versions, without, targets }
    }

    fn run(&self) -> Api {
        let items = self
            .public
            .items
            .iter()
            .filter_map(|(id, paths)| self.item(&self.krate.index[id], paths))
            .collect();
        let modules = self
            .public
            .modules
            .iter()
            .filter_map(|(id, path)| {
                let item = &self.krate.index[id];
                Some(ModuleDoc { path: path.clone(), docs: item.docs.clone()?, links: self.links(item) })
            })
            .collect();
        Api {
            krate: CRATE,
            version: self.krate.crate_version.clone().unwrap_or_default(),
            format_version: self.krate.format_version,
            features: API_FEATURES.to_vec(),
            modules,
            items,
        }
    }

    /// The features a key needs: those whose leave-one-out build has none of
    /// the item's paths followed by `suffix`.
    fn features(&self, paths: &[String], suffix: &str) -> Vec<&'static str> {
        self.without
            .iter()
            .filter(|(_, keys)| !paths.iter().any(|p| keys.contains(&format!("{p}{suffix}"))))
            .map(|(feature, _)| *feature)
            .collect()
    }

    fn item(&self, item: &Item, paths: &[String]) -> Option<ItemDoc> {
        let kind = match &item.inner {
            ItemEnum::Struct(_) => "struct",
            ItemEnum::Enum(_) => "enum",
            ItemEnum::Union(_) => "union",
            ItemEnum::Trait(_) => "trait",
            ItemEnum::Function(_) => "fn",
            ItemEnum::TypeAlias(_) => "type",
            ItemEnum::Constant { .. } => "const",
            ItemEnum::Static(_) => "static",
            ItemEnum::Macro(_) | ItemEnum::ProcMacro(_) => "macro",
            _ => return None,
        };
        let path = canonical(paths).to_string();
        let members = members(self.krate, item)
            .into_iter()
            .map(|member| MemberDoc {
                features: self.features(paths, &format!("::{}", member.key)),
                name: member.item.name.clone().unwrap_or_default(),
                section: member.section,
                signature: self.member_signature(member.item, member.section),
                docs: member.item.docs.clone().unwrap_or_default(),
                links: self.links(member.item),
                key: member.key,
            })
            .collect();
        let mut traits: Vec<TraitDoc> = trait_impls(self.krate, item)
            .into_iter()
            .filter(|t| !HIDDEN_TRAITS.contains(&last_segment(&t.path)))
            .map(|t| {
                let name = last_segment(&t.path).to_string();
                TraitDoc {
                    features: self.features(paths, &format!("::impl.{name}")),
                    href: self.href(&t.id),
                    name,
                }
            })
            .collect();
        traits.dedup_by(|a, b| a.name == b.name);
        Some(ItemDoc {
            aliases: paths.iter().filter(|p| **p != path).cloned().collect(),
            name: item.name.clone().unwrap_or_default(),
            kind,
            features: self.features(paths, ""),
            signature: self.item_signature(item, last_segment(&path)),
            docs: item.docs.clone().unwrap_or_default(),
            links: self.links(item),
            source: item.span.as_ref().map(|span| {
                format!("{}#L{}", span.filename.display(), span.begin.0)
            }),
            deprecated: item.deprecation.as_ref().map(|d| d.note.clone().unwrap_or_default()),
            members,
            traits,
            path,
        })
    }

    fn links(&self, item: &Item) -> BTreeMap<String, String> {
        item.links
            .iter()
            .filter_map(|(text, id)| Some((text.clone(), self.href(id)?)))
            .collect()
    }

    /// Where a link to an item goes: `api:` and its key for this crate's public
    /// items, its docs.rs page for another crate's, and nowhere for a private
    /// item.
    fn href(&self, id: &Id) -> Option<String> {
        if let Some(target) = self.targets.get(id) {
            return Some(target.clone());
        }
        let summary = self.krate.paths.get(id)?;
        if summary.crate_id == 0 {
            return None;
        }
        self.external_url(summary)
    }

    fn external_url(&self, summary: &ItemSummary) -> Option<String> {
        let name = &self.krate.external_crates.get(&summary.crate_id)?.name;
        let base = match name.as_str() {
            "std" | "core" | "alloc" => "https://doc.rust-lang.org/stable/".to_string(),
            _ => format!("https://docs.rs/{name}/{}/", self.versions.get(name)?),
        };
        let (last, dirs) = summary.path.split_last()?;
        let prefix = match summary.kind {
            ItemKind::Module => return Some(format!("{base}{}/index.html", summary.path.join("/"))),
            ItemKind::Struct => "struct",
            ItemKind::Enum => "enum",
            ItemKind::Union => "union",
            ItemKind::Trait => "trait",
            ItemKind::TraitAlias => "traitalias",
            ItemKind::Function => "fn",
            ItemKind::TypeAlias => "type",
            ItemKind::Constant => "constant",
            ItemKind::Static => "static",
            ItemKind::Macro => "macro",
            ItemKind::ProcDerive => "derive",
            ItemKind::ProcAttribute => "attr",
            ItemKind::Primitive => "primitive",
            _ => return None,
        };
        Some(format!("{base}{}/{prefix}.{last}.html", dirs.join("/")))
    }

    fn item_signature(&self, item: &Item, name: &str) -> Vec<Token> {
        let mut s = Sig::new(self);
        match &item.inner {
            ItemEnum::Struct(st) => {
                s.kw("pub struct ");
                s.name(name);
                s.generics(&st.generics);
                if let StructKind::Tuple(fields) = &st.kind {
                    s.text("(");
                    for (i, field) in fields.iter().enumerate() {
                        if i > 0 {
                            s.text(", ");
                        }
                        match field.and_then(|id| self.krate.index.get(&id)) {
                            Some(Item { inner: ItemEnum::StructField(ty), .. }) => {
                                s.kw("pub ");
                                s.ty(ty);
                            }
                            _ => s.text("/* private */"),
                        }
                    }
                    s.text(")");
                }
                s.where_clause(&st.generics);
            }
            ItemEnum::Enum(e) => {
                s.kw("pub enum ");
                s.name(name);
                s.generics(&e.generics);
                s.where_clause(&e.generics);
            }
            ItemEnum::Union(u) => {
                s.kw("pub union ");
                s.name(name);
                s.generics(&u.generics);
                s.where_clause(&u.generics);
            }
            ItemEnum::Trait(t) => {
                s.kw("pub trait ");
                s.name(name);
                s.generics(&t.generics);
                if !t.bounds.is_empty() {
                    s.text(": ");
                    s.bounds(&t.bounds);
                }
                s.where_clause(&t.generics);
            }
            ItemEnum::Function(f) => s.function(name, f, true),
            ItemEnum::TypeAlias(alias) => {
                s.kw("pub type ");
                s.name(name);
                s.generics(&alias.generics);
                s.text(" = ");
                s.ty(&alias.type_);
                s.where_clause(&alias.generics);
                s.text(";");
            }
            ItemEnum::Constant { type_, const_ } => {
                s.kw("pub const ");
                s.name(name);
                s.text(": ");
                s.ty(type_);
                let value = match (&const_.expr[..], &const_.value) {
                    ("_", Some(value)) => value.clone(),
                    (expr, _) => expr.to_string(),
                };
                s.text(&format!(" = {value};"));
            }
            ItemEnum::Static(st) => {
                s.kw(if st.is_mutable { "pub static mut " } else { "pub static " });
                s.name(name);
                s.text(": ");
                s.ty(&st.type_);
                s.text(";");
            }
            _ => s.name(name),
        }
        s.out
    }

    fn member_signature(&self, item: &Item, section: &str) -> Vec<Token> {
        let mut s = Sig::new(self);
        let name = item.name.as_deref().unwrap_or_default();
        match &item.inner {
            ItemEnum::StructField(ty) => {
                s.kw("pub ");
                s.name(name);
                s.text(": ");
                s.ty(ty);
            }
            ItemEnum::Variant(variant) => {
                s.name(name);
                match &variant.kind {
                    VariantKind::Plain => {}
                    VariantKind::Tuple(fields) => {
                        s.text("(");
                        for (i, field) in fields.iter().enumerate() {
                            if i > 0 {
                                s.text(", ");
                            }
                            match field.and_then(|id| self.krate.index.get(&id)) {
                                Some(Item { inner: ItemEnum::StructField(ty), .. }) => s.ty(ty),
                                _ => s.text("_"),
                            }
                        }
                        s.text(")");
                    }
                    VariantKind::Struct { fields, .. } => {
                        s.text(" { ");
                        for (i, field) in fields.iter().enumerate() {
                            if i > 0 {
                                s.text(", ");
                            }
                            let field = &self.krate.index[field];
                            s.text(field.name.as_deref().unwrap_or_default());
                            s.text(": ");
                            if let ItemEnum::StructField(ty) = &field.inner {
                                s.ty(ty);
                            }
                        }
                        s.text(" }");
                    }
                }
                if let Some(d) = &variant.discriminant {
                    s.text(&format!(" = {}", d.expr));
                }
            }
            ItemEnum::Function(f) => {
                let in_trait = section != "Methods";
                s.function(name, f, !in_trait && item.visibility == Visibility::Public);
            }
            ItemEnum::AssocType { generics, bounds, type_, .. } => {
                s.kw("type ");
                s.name(name);
                s.generics(generics);
                if !bounds.is_empty() {
                    s.text(": ");
                    s.bounds(bounds);
                }
                if let Some(ty) = type_ {
                    s.text(" = ");
                    s.ty(ty);
                }
                s.text(";");
            }
            ItemEnum::AssocConst { type_, value, .. } => {
                s.kw("const ");
                s.name(name);
                s.text(": ");
                s.ty(type_);
                if let Some(value) = value {
                    s.text(&format!(" = {value}"));
                }
                s.text(";");
            }
            _ => s.name(name),
        }
        s.out
    }
}

/// Writes a signature as tokens, linking each named type to its docs.
struct Sig<'a, 'b> {
    x: &'b Extractor<'a>,
    out: Vec<Token>,
}

impl<'a, 'b> Sig<'a, 'b> {
    fn new(x: &'b Extractor<'a>) -> Self {
        Sig { x, out: Vec::new() }
    }

    fn text(&mut self, text: &str) {
        if let Some(last) = self.out.last_mut() {
            if last.h.is_none() && last.k.is_none() {
                last.t.push_str(text);
                return;
            }
        }
        self.out.push(Token { t: text.into(), h: None, k: None });
    }

    fn kw(&mut self, text: &str) {
        self.out.push(Token { t: text.into(), h: None, k: Some("kw") });
    }

    fn name(&mut self, text: &str) {
        self.out.push(Token { t: text.into(), h: None, k: Some("name") });
    }

    fn link(&mut self, text: &str, id: &Id) {
        match self.x.href(id) {
            Some(href) => self.out.push(Token { t: text.into(), h: Some(href), k: None }),
            None => self.text(text),
        }
    }

    fn len(&self) -> usize {
        self.out.iter().map(|t| t.t.len()).sum()
    }

    fn path(&mut self, path: &Path) {
        self.link(last_segment(&path.path), &path.id);
        if let Some(args) = &path.args {
            self.generic_args(args);
        }
    }

    fn generic_args(&mut self, args: &GenericArgs) {
        match args {
            GenericArgs::AngleBracketed { args, constraints } => {
                if args.is_empty() && constraints.is_empty() {
                    return;
                }
                self.text("<");
                let mut first = true;
                for arg in args {
                    if !std::mem::take(&mut first) {
                        self.text(", ");
                    }
                    match arg {
                        GenericArg::Lifetime(l) => self.text(l),
                        GenericArg::Type(ty) => self.ty(ty),
                        GenericArg::Const(c) => self.text(&c.expr),
                        GenericArg::Infer => self.text("_"),
                    }
                }
                for constraint in constraints {
                    if !std::mem::take(&mut first) {
                        self.text(", ");
                    }
                    self.text(&constraint.name);
                    if let Some(args) = &constraint.args {
                        self.generic_args(args);
                    }
                    match &constraint.binding {
                        rustdoc_types::AssocItemConstraintKind::Equality(term) => {
                            self.text(" = ");
                            self.term(term);
                        }
                        rustdoc_types::AssocItemConstraintKind::Constraint(bounds) => {
                            self.text(": ");
                            self.bounds(bounds);
                        }
                    }
                }
                self.text(">");
            }
            GenericArgs::Parenthesized { inputs, output } => {
                self.text("(");
                for (i, input) in inputs.iter().enumerate() {
                    if i > 0 {
                        self.text(", ");
                    }
                    self.ty(input);
                }
                self.text(")");
                if let Some(output) = output {
                    self.text(" -> ");
                    self.ty(output);
                }
            }
            GenericArgs::ReturnTypeNotation => self.text("(..)"),
        }
    }

    fn term(&mut self, term: &Term) {
        match term {
            Term::Type(ty) => self.ty(ty),
            Term::Constant(c) => self.text(&c.expr),
        }
    }

    fn ty(&mut self, ty: &Type) {
        match ty {
            Type::ResolvedPath(path) => self.path(path),
            Type::DynTrait(dyn_trait) => {
                self.kw("dyn ");
                for (i, poly) in dyn_trait.traits.iter().enumerate() {
                    if i > 0 {
                        self.text(" + ");
                    }
                    self.hrtb(&poly.generic_params);
                    self.path(&poly.trait_);
                }
                if let Some(lifetime) = &dyn_trait.lifetime {
                    self.text(&format!(" + {lifetime}"));
                }
            }
            Type::Generic(name) | Type::Primitive(name) => self.text(name),
            Type::FunctionPointer(f) => {
                self.hrtb(&f.generic_params);
                self.kw("fn");
                self.text("(");
                for (i, (_, input)) in f.sig.inputs.iter().enumerate() {
                    if i > 0 {
                        self.text(", ");
                    }
                    self.ty(input);
                }
                self.text(")");
                if let Some(output) = &f.sig.output {
                    self.text(" -> ");
                    self.ty(output);
                }
            }
            Type::Tuple(types) => {
                self.text("(");
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        self.text(", ");
                    }
                    self.ty(ty);
                }
                self.text(if types.len() == 1 { ",)" } else { ")" });
            }
            Type::Slice(ty) => {
                self.text("[");
                self.ty(ty);
                self.text("]");
            }
            Type::Array { type_, len } => {
                self.text("[");
                self.ty(type_);
                self.text(&format!("; {len}]"));
            }
            Type::Pat { type_, .. } => self.ty(type_),
            Type::ImplTrait(bounds) => {
                self.kw("impl ");
                self.bounds(bounds);
            }
            Type::Infer => self.text("_"),
            Type::RawPointer { is_mutable, type_ } => {
                self.text(if *is_mutable { "*mut " } else { "*const " });
                self.ty(type_);
            }
            Type::BorrowedRef { lifetime, is_mutable, type_ } => {
                self.text("&");
                if let Some(lifetime) = lifetime {
                    self.text(&format!("{lifetime} "));
                }
                if *is_mutable {
                    self.kw("mut ");
                }
                self.ty(type_);
            }
            Type::QualifiedPath { name, args, self_type, trait_ } => {
                match (trait_, &**self_type) {
                    (Some(trait_), self_type) if !matches!(self_type, Type::Generic(_)) => {
                        self.text("<");
                        self.ty(self_type);
                        self.kw(" as ");
                        self.path(trait_);
                        self.text(">");
                    }
                    _ => self.ty(self_type),
                }
                self.text(&format!("::{name}"));
                if let Some(args) = args {
                    self.generic_args(args);
                }
            }
        }
    }

    fn hrtb(&mut self, params: &[GenericParamDef]) {
        if !params.is_empty() {
            let names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
            self.text(&format!("for<{}> ", names.join(", ")));
        }
    }

    fn bounds(&mut self, bounds: &[GenericBound]) {
        for (i, bound) in bounds.iter().enumerate() {
            if i > 0 {
                self.text(" + ");
            }
            match bound {
                GenericBound::TraitBound { trait_, generic_params, modifier } => {
                    match modifier {
                        TraitBoundModifier::None => {}
                        TraitBoundModifier::Maybe => self.text("?"),
                        TraitBoundModifier::MaybeConst => self.text("~const "),
                    }
                    self.hrtb(generic_params);
                    self.path(trait_);
                }
                GenericBound::Outlives(lifetime) => self.text(lifetime),
                GenericBound::Use(args) => {
                    let names: Vec<&str> = args
                        .iter()
                        .map(|a| match a {
                            rustdoc_types::PreciseCapturingArg::Lifetime(n)
                            | rustdoc_types::PreciseCapturingArg::Param(n) => n.as_str(),
                        })
                        .collect();
                    self.text(&format!("use<{}>", names.join(", ")));
                }
            }
        }
    }

    fn generics(&mut self, generics: &Generics) {
        let params: Vec<&GenericParamDef> = generics
            .params
            .iter()
            .filter(|p| !matches!(p.kind, GenericParamDefKind::Type { is_synthetic: true, .. }))
            .collect();
        if params.is_empty() {
            return;
        }
        self.text("<");
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.text(", ");
            }
            match &param.kind {
                GenericParamDefKind::Lifetime { outlives } => {
                    self.text(&param.name);
                    if !outlives.is_empty() {
                        self.text(&format!(": {}", outlives.join(" + ")));
                    }
                }
                GenericParamDefKind::Type { bounds, default, .. } => {
                    self.text(&param.name);
                    if !bounds.is_empty() {
                        self.text(": ");
                        self.bounds(bounds);
                    }
                    if let Some(default) = default {
                        self.text(" = ");
                        self.ty(default);
                    }
                }
                GenericParamDefKind::Const { type_, default } => {
                    self.kw("const ");
                    self.text(&format!("{}: ", param.name));
                    self.ty(type_);
                    if let Some(default) = default {
                        self.text(&format!(" = {default}"));
                    }
                }
            }
        }
        self.text(">");
    }

    fn where_clause(&mut self, generics: &Generics) {
        if generics.where_predicates.is_empty() {
            return;
        }
        self.kw("\nwhere\n");
        for predicate in &generics.where_predicates {
            self.text("    ");
            match predicate {
                WherePredicate::BoundPredicate { type_, bounds, generic_params } => {
                    self.hrtb(generic_params);
                    self.ty(type_);
                    self.text(": ");
                    self.bounds(bounds);
                }
                WherePredicate::LifetimePredicate { lifetime, outlives } => {
                    self.text(&format!("{lifetime}: {}", outlives.join(" + ")));
                }
                WherePredicate::EqPredicate { lhs, rhs } => {
                    self.ty(lhs);
                    self.text(" = ");
                    self.term(rhs);
                }
            }
            self.text(",\n");
        }
    }

    fn function(&mut self, name: &str, f: &Function, public: bool) {
        if public {
            self.kw("pub ");
        }
        if f.header.is_const {
            self.kw("const ");
        }
        if f.header.is_async {
            self.kw("async ");
        }
        if f.header.is_unsafe {
            self.kw("unsafe ");
        }
        self.kw("fn ");
        self.name(name);
        self.generics(&f.generics);

        // Long argument lists go one per line, as rustfmt would write them.
        let inputs: Vec<Vec<Token>> = f
            .sig
            .inputs
            .iter()
            .map(|(name, ty)| {
                let mut arg = Sig::new(self.x);
                arg.input(name, ty);
                arg.out
            })
            .collect();
        let width: usize = self.len() + inputs.iter().flatten().map(|t| t.t.len() + 2).sum::<usize>();
        let wrap = width > 80;
        self.text("(");
        for (i, input) in inputs.into_iter().enumerate() {
            if wrap {
                self.text("\n    ");
            } else if i > 0 {
                self.text(", ");
            }
            for token in input {
                match (token.h, token.k) {
                    (None, None) => self.text(&token.t),
                    (h, k) => self.out.push(Token { t: token.t, h, k }),
                }
            }
            if wrap {
                self.text(",");
            }
        }
        self.text(if wrap { "\n)" } else { ")" });
        if let Some(output) = &f.sig.output {
            self.text(" -> ");
            self.ty(output);
        }
        self.where_clause(&f.generics);
    }

    fn input(&mut self, name: &str, ty: &Type) {
        if name == "self" {
            match ty {
                Type::Generic(s) if s == "Self" => return self.kw("self"),
                Type::BorrowedRef { lifetime, is_mutable, type_ }
                    if matches!(&**type_, Type::Generic(s) if s == "Self") =>
                {
                    self.text("&");
                    if let Some(lifetime) = lifetime {
                        self.text(&format!("{lifetime} "));
                    }
                    return self.kw(if *is_mutable { "mut self" } else { "self" });
                }
                _ => {}
            }
        }
        self.text(&format!("{name}: "));
        self.ty(ty);
    }
}
