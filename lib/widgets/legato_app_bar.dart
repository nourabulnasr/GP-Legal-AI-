import 'package:flutter/material.dart';

/// App bar with an explicit back button whenever this route can pop.
class LegatoAppBar extends StatelessWidget implements PreferredSizeWidget {
  const LegatoAppBar({
    super.key,
    required this.title,
    this.actions,
    this.bottom,
    this.centerTitle,
    this.leading,
  });

  final Widget title;
  final List<Widget>? actions;
  final PreferredSizeWidget? bottom;
  final bool? centerTitle;
  final Widget? leading;

  @override
  Size get preferredSize {
    final bottomHeight = bottom?.preferredSize.height ?? 0;
    return Size.fromHeight(kToolbarHeight + bottomHeight);
  }

  @override
  Widget build(BuildContext context) {
    final canPop = Navigator.of(context).canPop();
    return AppBar(
      title: title,
      actions: actions,
      bottom: bottom,
      centerTitle: centerTitle,
      automaticallyImplyLeading: false,
      leading: leading ??
          (canPop
              ? IconButton(
                  icon: const Icon(Icons.arrow_back),
                  tooltip: MaterialLocalizations.of(context).backButtonTooltip,
                  onPressed: () => Navigator.of(context).maybePop(),
                )
              : null),
    );
  }
}

/// Standard page shell for secondary routes (title + back + body).
class LegatoPageScaffold extends StatelessWidget {
  const LegatoPageScaffold({
    super.key,
    required this.title,
    required this.body,
    this.actions,
    this.bottom,
  });

  final String title;
  final Widget body;
  final List<Widget>? actions;
  final PreferredSizeWidget? bottom;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: LegatoAppBar(
        title: Text(title),
        actions: actions,
        bottom: bottom,
      ),
      body: body,
    );
  }
}
