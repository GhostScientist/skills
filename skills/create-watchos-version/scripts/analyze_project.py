#!/usr/bin/env python3
"""
Analyze an Apple platform project structure for watchOS compatibility.
Outputs a structured report of project architecture, dependencies, and potential issues.

Usage: python3 analyze_project.py <project_path>
"""

import os
import sys
import json
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

@dataclass
class ProjectAnalysis:
    """Results of project analysis."""
    project_path: str
    project_type: str = "unknown"  # xcodeproj, swiftpm, workspace
    app_name: str = ""
    
    # Architecture
    ui_framework: str = "unknown"  # swiftui, uikit, appkit, hybrid
    data_layer: list = field(default_factory=list)
    state_management: list = field(default_factory=list)
    
    # Dependencies
    spm_packages: list = field(default_factory=list)
    cocoapods: bool = False
    carthage: bool = False
    
    # Existing targets
    has_watch_target: bool = False
    has_widget_extension: bool = False
    ios_deployment_target: str = ""
    
    # Code patterns found
    frameworks_used: list = field(default_factory=list)
    watchos_incompatible: list = field(default_factory=list)
    
    # Files
    swift_files: int = 0
    view_files: list = field(default_factory=list)
    model_files: list = field(default_factory=list)
    service_files: list = field(default_factory=list)


def find_project_files(root: Path) -> dict:
    """Find key project configuration files."""
    files = {
        "xcodeproj": None,
        "xcworkspace": None,
        "package_swift": None,
        "podfile": None,
        "cartfile": None,
        "info_plists": [],
        "swift_files": [],
        "pbxproj": None,
    }
    
    for path in root.rglob("*"):
        if path.is_file():
            name = path.name.lower()
            if name.endswith(".xcodeproj"):
                files["xcodeproj"] = path
            elif name.endswith(".xcworkspace"):
                files["xcworkspace"] = path
            elif name == "package.swift":
                files["package_swift"] = path
            elif name == "podfile":
                files["podfile"] = path
            elif name == "cartfile":
                files["cartfile"] = path
            elif name == "info.plist":
                files["info_plists"].append(path)
            elif name.endswith(".swift"):
                files["swift_files"].append(path)
            elif name == "project.pbxproj":
                files["pbxproj"] = path
                
    return files


def analyze_swift_file(path: Path) -> dict:
    """Analyze a Swift file for patterns and frameworks."""
    result = {
        "frameworks": set(),
        "is_swiftui_view": False,
        "is_uikit": False,
        "is_model": False,
        "is_viewmodel": False,
        "is_service": False,
        "watchos_issues": [],
    }
    
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except:
        return result
    
    # Framework imports
    import_pattern = r'^import\s+(\w+)'
    for match in re.finditer(import_pattern, content, re.MULTILINE):
        result["frameworks"].add(match.group(1))
    
    # SwiftUI detection
    if "import SwiftUI" in content:
        if ": View" in content or "var body: some View" in content:
            result["is_swiftui_view"] = True
    
    # UIKit detection
    if "import UIKit" in content:
        result["is_uikit"] = True
        if "UIViewController" in content or "UIView" in content:
            result["watchos_issues"].append("UIKit UIViewController/UIView")
    
    # Model detection
    name_lower = path.stem.lower()
    if any(x in name_lower for x in ["model", "entity", "dto"]):
        result["is_model"] = True
    if "@Model" in content:  # SwiftData
        result["is_model"] = True
    if "NSManagedObject" in content:  # Core Data
        result["is_model"] = True
        
    # ViewModel detection
    if "viewmodel" in name_lower or "ObservableObject" in content:
        result["is_viewmodel"] = True
        
    # Service detection
    if any(x in name_lower for x in ["service", "manager", "repository", "api", "client"]):
        result["is_service"] = True
    
    # watchOS incompatible patterns
    watchos_blockers = [
        ("WKWebView", "WebKit not available on watchOS"),
        ("UITableView", "Use SwiftUI List instead"),
        ("UICollectionView", "Use SwiftUI LazyVGrid/LazyHGrid"),
        ("ARSession", "ARKit not available on watchOS"),
        ("import WebKit", "WebKit not available on watchOS"),
        ("import ARKit", "ARKit not available on watchOS"),
        ("import Vision", "Vision limited on watchOS"),
        ("AVPlayer", "Video playback not supported on watchOS"),
        ("MKMapView", "Use limited MapKit or link to iPhone"),
    ]
    
    for pattern, issue in watchos_blockers:
        if pattern in content:
            result["watchos_issues"].append(issue)
    
    return result


def analyze_pbxproj(path: Path) -> dict:
    """Parse pbxproj for deployment targets and targets."""
    result = {
        "ios_deployment": "",
        "watchos_deployment": "",
        "has_watch_target": False,
        "has_widget": False,
        "targets": [],
    }
    
    if not path or not path.exists():
        return result
    
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except:
        return result
    
    # Deployment targets
    ios_match = re.search(r'IPHONEOS_DEPLOYMENT_TARGET\s*=\s*(\d+\.?\d*)', content)
    if ios_match:
        result["ios_deployment"] = ios_match.group(1)
        
    watch_match = re.search(r'WATCHOS_DEPLOYMENT_TARGET\s*=\s*(\d+\.?\d*)', content)
    if watch_match:
        result["watchos_deployment"] = watch_match.group(1)
        result["has_watch_target"] = True
    
    # Watch target detection
    if "watchOS" in content or "WatchKit" in content:
        result["has_watch_target"] = True
        
    # Widget extension
    if "WidgetKit" in content or "Widget Extension" in content:
        result["has_widget"] = True
    
    return result


def analyze_package_swift(path: Path) -> list:
    """Extract SPM dependencies from Package.swift."""
    packages = []
    
    if not path or not path.exists():
        return packages
    
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except:
        return packages
    
    # Simple pattern matching for package URLs
    url_pattern = r'\.package\s*\(\s*url:\s*"([^"]+)"'
    for match in re.finditer(url_pattern, content):
        url = match.group(1)
        # Extract package name from URL
        name = url.rstrip("/").split("/")[-1].replace(".git", "")
        packages.append({"name": name, "url": url})
    
    return packages


def analyze_project(project_path: str) -> ProjectAnalysis:
    """Main analysis function."""
    root = Path(project_path).resolve()
    
    if not root.exists():
        print(f"Error: Path does not exist: {root}", file=sys.stderr)
        sys.exit(1)
    
    analysis = ProjectAnalysis(project_path=str(root))
    files = find_project_files(root)
    
    # Determine project type
    if files["xcworkspace"]:
        analysis.project_type = "xcworkspace"
    elif files["xcodeproj"]:
        analysis.project_type = "xcodeproj"
    elif files["package_swift"]:
        analysis.project_type = "swiftpm"
    
    # Check for dependency managers
    analysis.cocoapods = files["podfile"] is not None
    analysis.carthage = files["cartfile"] is not None
    
    # Analyze pbxproj
    pbx_info = analyze_pbxproj(files["pbxproj"])
    analysis.ios_deployment_target = pbx_info["ios_deployment"]
    analysis.has_watch_target = pbx_info["has_watch_target"]
    analysis.has_widget_extension = pbx_info["has_widget"]
    
    # Analyze SPM packages
    analysis.spm_packages = analyze_package_swift(files["package_swift"])
    
    # Analyze Swift files
    all_frameworks = set()
    all_issues = set()
    swiftui_views = 0
    uikit_usage = 0
    
    analysis.swift_files = len(files["swift_files"])
    
    for swift_file in files["swift_files"]:
        file_analysis = analyze_swift_file(swift_file)
        all_frameworks.update(file_analysis["frameworks"])
        all_issues.update(file_analysis["watchos_issues"])
        
        rel_path = str(swift_file.relative_to(root))
        
        if file_analysis["is_swiftui_view"]:
            swiftui_views += 1
            analysis.view_files.append(rel_path)
        if file_analysis["is_uikit"]:
            uikit_usage += 1
        if file_analysis["is_model"]:
            analysis.model_files.append(rel_path)
        if file_analysis["is_service"]:
            analysis.service_files.append(rel_path)
    
    # Determine UI framework
    if swiftui_views > 0 and uikit_usage > 0:
        analysis.ui_framework = "hybrid"
    elif swiftui_views > 0:
        analysis.ui_framework = "swiftui"
    elif uikit_usage > 0:
        analysis.ui_framework = "uikit"
    
    # Data layer detection
    if "SwiftData" in all_frameworks:
        analysis.data_layer.append("SwiftData")
    if "CoreData" in all_frameworks:
        analysis.data_layer.append("CoreData")
    if "RealmSwift" in all_frameworks or "Realm" in all_frameworks:
        analysis.data_layer.append("Realm")
    
    # State management
    if "Combine" in all_frameworks:
        analysis.state_management.append("Combine")
    if "ComposableArchitecture" in all_frameworks:
        analysis.state_management.append("TCA")
    
    analysis.frameworks_used = sorted(list(all_frameworks))
    analysis.watchos_incompatible = sorted(list(all_issues))
    
    # Limit file lists to avoid huge output
    analysis.view_files = analysis.view_files[:20]
    analysis.model_files = analysis.model_files[:20]
    analysis.service_files = analysis.service_files[:20]
    
    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_project.py <project_path>")
        print("\nAnalyzes an Apple platform project for watchOS compatibility.")
        sys.exit(1)
    
    project_path = sys.argv[1]
    analysis = analyze_project(project_path)
    
    # Output as JSON for easy parsing
    print(json.dumps(asdict(analysis), indent=2))


if __name__ == "__main__":
    main()
